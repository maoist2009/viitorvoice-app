/* qnn_executor.cpp — in-process QNN HTP executor.
 *
 * Ported from rvc_npu/qnn_resident.c (the resident multi-context daemon that
 * ran the full 5-stage RVC pipeline bit-exact against qnn-net-run). The daemon
 * framing (u32-length records over stdin/stdout) is replaced by direct C entry
 * points; the QNN sequence and the pinned 4 KB-aligned client buffers are
 * unchanged, because the HTP context binds clientBuf.data across graphExecute
 * calls and the buffers must never move once the context is created.
 */
#include "qnn_executor.h"

#include <dlfcn.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <android/log.h>

#include "QnnInterface.h"
#include "System/QnnSystemContext.h"
#include "System/QnnSystemDlc.h"
#include "System/QnnSystemInterface.h"
#include "HTP/QnnHtpDevice.h"
#include "HTP/QnnHtpPerfInfrastructure.h"

#define TAG "vv_native"

#define MAX_GRAPHS  28
#define MAX_TENSORS 16
#define NAME_LEN    128
#define ERR_LEN     1024

static const QnnInterface_t *IFACE;   /* -> v2_34 */
static Qnn_LogHandle_t      LOG;
static Qnn_BackendHandle_t  BACKEND;
static Qnn_DeviceHandle_t   DEVICE;
static int                  INIT_DONE = 0;
static pthread_mutex_t      LOCK = PTHREAD_MUTEX_INITIALIZER;
static char                 ERR_BUF[ERR_LEN];

static void set_error(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(ERR_BUF, sizeof(ERR_BUF), fmt, ap);
  va_end(ap);
}

const char *qnn_last_error(void) { return ERR_BUF; }

/* Matches qnn_resident's android log path; logs go to logcat under rvc_native. */
static void qlog(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  __android_log_vprint(ANDROID_LOG_INFO, TAG, fmt, ap);
  va_end(ap);
}

static void qlog_cb(const char *fmt, QnnLog_Level_t level, uint64_t ts, va_list ap) {
  char buf[1024];
  int n = snprintf(buf, sizeof(buf), "[qnn:%d ts=%llu] ", (int)level,
                   (unsigned long long)ts);
  if (n > 0 && (size_t)n < sizeof(buf))
    vsnprintf(buf + n, sizeof(buf) - n, fmt, ap);
  __android_log_print(ANDROID_LOG_INFO, TAG, "%s", buf);
}

/* --- per-graph state ----------------------------------------------------- */

struct tensor {
  char     name[NAME_LEN];
  uint32_t id;       /* HTP matches graph tensors by id, not name */
  uint32_t rank, dims[8];
  Qnn_DataType_t     dtype;
  Qnn_TensorDataFormat_t dataFormat;
  Qnn_QuantizeParams_t   quantizeParams;
  size_t   bytes;
  void    *buf;      /* 4 KB-aligned, stable for the context's lifetime */
};

/* A multi-graph context binary shares one mmap and one HTP context between
 * several graphs (llm_256 + llm_512 live in the same file). The resources are
 * reference counted and released with the last graph that uses them. */
struct grp {
  int      used;
  int      refs;
  int      fd;
  void    *map;
  size_t   map_len;
  Qnn_ContextHandle_t ctx;
};

static struct grp GRPS[4];

struct graph {
  int      used;
  char     tag[NAME_LEN];
  int      fd;
  void    *map;
  size_t   map_len;
  struct grp *grp;             /* nonzero when the graph lives in a multi-graph bin */
  QnnSystemDlc_Handle_t dlc;   /* nonzero when loaded from a .dlc file */
  Qnn_ContextHandle_t ctx;
  Qnn_GraphHandle_t    g;
  uint32_t n_in, n_out;
  struct tensor in[MAX_TENSORS];
  struct tensor out[MAX_TENSORS];
  /* graphExecute-only wall time, excluding host marshalling. */
  uint64_t exec_ns_sum = 0;
  uint64_t exec_ns_min = 0;
  uint64_t exec_ns_max = 0;
  uint64_t exec_n = 0;
};

static struct graph GRAPHS[MAX_GRAPHS];

static struct graph *find_graph(const char *tag) {
  for (int i = 0; i < MAX_GRAPHS; i++)
    if (GRAPHS[i].used && !strcmp(GRAPHS[i].tag, tag)) return &GRAPHS[i];
  return NULL;
}

static size_t dtype_bytes(Qnn_DataType_t dt, size_t elems) {
  switch (dt) {
    case QNN_DATATYPE_FLOAT_32:
    case QNN_DATATYPE_INT_32:
    case QNN_DATATYPE_UINT_32:
      return elems * 4;
    case QNN_DATATYPE_FLOAT_16:
      return elems * 2;
    default:
      return 0;   /* unsupported at the boundary — refuse, never miscast */
  }
}

/* fp16 <-> fp32 at the graph boundary. An ONNX exported in fp16 keeps fp16
 * graph IO; the older fp32 exports had fp32 IO with internal casts. */
static inline float h2f(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000) << 16;
  uint32_t exp = (h >> 10) & 0x1F;
  uint32_t man = h & 0x3FF;
  uint32_t out;
  if (exp == 0) {
    if (man == 0) {
      out = sign;
    } else {
      exp = 127 - 15 + 1;
      while ((man & 0x400) == 0) { man <<= 1; exp--; }
      man &= 0x3FF;
      out = sign | (exp << 23) | (man << 13);
    }
  } else if (exp == 31) {
    out = sign | 0x7F800000u | (man << 13);
  } else {
    out = sign | ((exp + 127 - 15) << 23) | (man << 13);
  }
  float f;
  memcpy(&f, &out, 4);
  return f;
}

static inline uint16_t f2h(float f) {
  uint32_t x;
  memcpy(&x, &f, 4);
  uint32_t sign = (x >> 16) & 0x8000;
  uint32_t e = (x >> 23) & 0xFF;
  uint32_t man = x & 0x7FFFFF;
  if (e == 0xFF) return (uint16_t)(sign | 0x7C00 | (man ? 0x200 : 0));
  int32_t exp = (int32_t)e - 127 + 15;
  if (exp >= 31) return (uint16_t)(sign | 0x7C00);
  if (exp <= 0) {
    if (exp < -10) return (uint16_t)sign;
    uint32_t m = (man | 0x800000) >> (uint32_t)(14 - exp);
    if (((man | 0x800000) >> (uint32_t)(13 - exp)) & 1) m++;
    return (uint16_t)(sign | m);
  }
  uint32_t r = (uint32_t)exp << 10 | (man >> 13);
  if (man & 0x1000) r++;
  return (uint16_t)(sign | r);
}

static void free_graph(struct graph *g) {
#define IF IFACE->v2_34
  if (g->g) IF.graphFinalize(g->g, NULL, NULL);
  if (g->grp) {
    /* multi-graph bin: the context, mmap and fd belong to the group */
    if (--g->grp->refs <= 0) {
      if (g->grp->ctx) IF.contextFree(g->grp->ctx, NULL);
      if (g->grp->map && g->grp->map != MAP_FAILED)
        munmap(g->grp->map, g->grp->map_len);
      if (g->grp->fd >= 0) close(g->grp->fd);
      memset(g->grp, 0, sizeof(*g->grp));
      g->grp->fd = -1;
    }
    g->grp = NULL;
  } else if (g->ctx) {
    IF.contextFree(g->ctx, NULL);
  }
#undef IF
  if (g->dlc) {
    /* the DLC owns the weight memory the context's graphs reference, so it
     * must outlive the contextFree above */
    void *sh = dlopen("libQnnSystem.so", RTLD_NOW);
    if (sh) {
      Qnn_ErrorHandle_t (*sysGetProviders)(const QnnSystemInterface_t ***, uint32_t *) =
          (Qnn_ErrorHandle_t(*)(const QnnSystemInterface_t ***, uint32_t *))
              dlsym(sh, "QnnSystemInterface_getProviders");
      const QnnSystemInterface_t **list = NULL;
      uint32_t n = 0;
      if (sysGetProviders && !sysGetProviders(&list, &n)) {
        for (uint32_t i = 0; i < n; i++)
          if (list[i]->systemApiVersion.major == QNN_SYSTEM_API_VERSION_MAJOR &&
              list[i]->systemApiVersion.minor >= QNN_SYSTEM_API_VERSION_MINOR &&
              list[i]->v1_9.systemDlcFree) {
            list[i]->v1_9.systemDlcFree(g->dlc);
            break;
          }
      }
    }
    g->dlc = 0;
  }
  if (g->map && g->map != MAP_FAILED) munmap(g->map, g->map_len);
  if (g->fd >= 0) close(g->fd);
  for (uint32_t i = 0; i < g->n_in; i++) free(g->in[i].buf);
  for (uint32_t i = 0; i < g->n_out; i++) free(g->out[i].buf);
  memset(g, 0, sizeof(*g));
  g->fd = -1;
}

/* Copy a tensor descriptor out of the system metadata into a plain struct.
 * Mirrors qnn_min.c's tensor_v2_from_sys. */
static int describe(const Qnn_Tensor_t *src, struct tensor *t) {
  memset(t, 0, sizeof(*t));
  const char *name = NULL;
  uint32_t rank = 0;
  const uint32_t *dims = NULL;
  if (src->version == QNN_TENSOR_VERSION_2) {
    name = src->v2.name;
    rank = src->v2.rank;
    dims = src->v2.dimensions;
    t->dtype = src->v2.dataType;
    t->id = src->v2.id;
    t->dataFormat = src->v2.dataFormat;
    t->quantizeParams = src->v2.quantizeParams;
  } else {
    name = src->v1.name;
    rank = src->v1.rank;
    dims = src->v1.dimensions;
    t->dtype = src->v1.dataType;
    t->id = src->v1.id;
    t->dataFormat = src->v1.dataFormat;
    t->quantizeParams = src->v1.quantizeParams;
  }
  if (!name) {
    set_error("tensor name null (v%d)", (int)src->version);
    return -1;
  }
  if (strlen(name) >= NAME_LEN) {
    set_error("tensor name too long (%.24s...)", name);
    return -1;
  }
  if (rank > 8) {
    set_error("tensor '%s' rank %u", name, rank);
    return -1;
  }
  strcpy(t->name, name);
  t->rank = rank;
  size_t elems = 1;
  for (uint32_t i = 0; i < rank; i++) {
    t->dims[i] = dims[i];
    elems *= dims[i];
  }
  size_t b = dtype_bytes(t->dtype, elems);
  if (!b) {
    set_error("tensor '%s' dtype %d ver %d rank %u", name, (int)t->dtype,
              (int)src->version, rank);
    return -1;
  }
  t->bytes = b;
  void *p = NULL;
  if (posix_memalign(&p, 4096, b) != 0 || !p) return -1;
  t->buf = p;
  return 0;
}

/* libQnnSystem metadata: every graph in the binary, with tensor descriptors. */
static int graph_infos(const void *bin, size_t len,
                       const QnnSystemContext_GraphInfo_t **graphs_out,
                       uint32_t *count_out) {
  void *sh = dlopen("libQnnSystem.so", RTLD_NOW);
  if (!sh) {
    set_error("dlopen libQnnSystem.so: %s", dlerror());
    return -1;
  }
  Qnn_ErrorHandle_t (*sysGetProviders)(const QnnSystemInterface_t ***, uint32_t *) =
      (Qnn_ErrorHandle_t(*)(const QnnSystemInterface_t ***, uint32_t *))
          dlsym(sh, "QnnSystemInterface_getProviders");
  if (!sysGetProviders) {
    set_error("dlsym QnnSystemInterface_getProviders: %s", dlerror());
    return -1;
  }
  const QnnSystemInterface_t **list = NULL;
  uint32_t n = 0;
  if (sysGetProviders(&list, &n)) return -1;
  const QnnSystemInterface_t *sel = NULL;
  for (uint32_t i = 0; i < n; i++)
    if (list[i]->systemApiVersion.major == QNN_SYSTEM_API_VERSION_MAJOR &&
        list[i]->systemApiVersion.minor >= QNN_SYSTEM_API_VERSION_MINOR)
      sel = list[i];
  if (!sel) {
    set_error("no QNN System API %u.%u provider",
              QNN_SYSTEM_API_VERSION_MAJOR, QNN_SYSTEM_API_VERSION_MINOR);
    return -1;
  }
#define SIF sel->v1_9
  QnnSystemContext_Handle_t sysCtx = NULL;
  if (SIF.systemContextCreate(&sysCtx)) return -1;
  const QnnSystemContext_BinaryInfo_t *info = NULL;
  Qnn_ContextBinarySize_t infoSize = 0;
  if (SIF.systemContextGetBinaryInfo(sysCtx, (void *)bin, (uint64_t)len, &info,
                                     &infoSize)) {
    set_error("systemContextGetBinaryInfo failed");
    return -1;
  }
  if (!info) return -1;
  uint32_t numGraphs = 0;
  const QnnSystemContext_GraphInfo_t *graphs = NULL;
  switch (info->version) {
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1:
      numGraphs = info->contextBinaryInfoV1.numGraphs;
      graphs = info->contextBinaryInfoV1.graphs;
      break;
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2:
      numGraphs = info->contextBinaryInfoV2.numGraphs;
      graphs = info->contextBinaryInfoV2.graphs;
      break;
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3:
      numGraphs = info->contextBinaryInfoV3.numGraphs;
      graphs = info->contextBinaryInfoV3.graphs;
      break;
    default:
      set_error("unknown BinaryInfo version %d", (int)info->version);
      return -1;
  }
  if (numGraphs == 0 || !graphs) {
    set_error("no graphs in context binary");
    return -1;
  }
  *graphs_out = graphs;
  *count_out = numGraphs;
  return 0;
#undef SIF
}

int qnn_init(const char *lib_dir, const char *skel_dir) {
  pthread_mutex_lock(&LOCK);
  if (INIT_DONE) {
    pthread_mutex_unlock(&LOCK);
    return 0;
  }
  /* Under the bionic linker only ADSP_LIBRARY_PATH decides where the HTP stub
   * looks for libQnnHtpV79Skel.so; LD_LIBRARY_PATH is set as well because that
   * is what the Termux-side path relies on. Both must be set before the first
   * call into libQnnHtp.so. */
  setenv("ADSP_LIBRARY_PATH", skel_dir, 1);
  setenv("LD_LIBRARY_PATH", lib_dir, 1);

  void *h = dlopen("libQnnHtp.so", RTLD_NOW);
  if (!h) {
    set_error("dlopen libQnnHtp.so: %s", dlerror());
    pthread_mutex_unlock(&LOCK);
    return -1;
  }
  Qnn_ErrorHandle_t (*getProviders)(const QnnInterface_t ***, uint32_t *) =
      (Qnn_ErrorHandle_t(*)(const QnnInterface_t ***, uint32_t *))
          dlsym(h, "QnnInterface_getProviders");
  if (!getProviders) {
    set_error("dlsym QnnInterface_getProviders: %s", dlerror());
    pthread_mutex_unlock(&LOCK);
    return -1;
  }
  const QnnInterface_t **providers = NULL;
  uint32_t num = 0;
  if (getProviders(&providers, &num)) {
    set_error("QnnInterface_getProviders failed");
    pthread_mutex_unlock(&LOCK);
    return -1;
  }
  for (uint32_t i = 0; i < num; i++) {
    const Qnn_ApiVersion_t *v = &providers[i]->apiVersion;
    qlog("provider[%u] %s core=%u.%u.%u backend=%u.%u.%u", i,
         providers[i]->providerName ? providers[i]->providerName : "(null)",
         v->coreApiVersion.major, v->coreApiVersion.minor, v->coreApiVersion.patch,
         v->backendApiVersion.major, v->backendApiVersion.minor,
         v->backendApiVersion.patch);
    if (v->coreApiVersion.major == QNN_API_VERSION_MAJOR &&
        v->coreApiVersion.minor >= QNN_API_VERSION_MINOR)
      IFACE = providers[i];
  }
  if (!IFACE) {
    set_error("no QNN API %u.%u provider", QNN_API_VERSION_MAJOR,
              QNN_API_VERSION_MINOR);
    pthread_mutex_unlock(&LOCK);
    return -1;
  }
#define IF IFACE->v2_34
  Qnn_ErrorHandle_t e;
  e = IF.logCreate(qlog_cb, QNN_LOG_LEVEL_DEBUG, &LOG);
  if (e) {
    set_error("logCreate 0x%llx", (unsigned long long)e);
    pthread_mutex_unlock(&LOCK);
    return -1;
  }
  e = IF.backendCreate(LOG, NULL, &BACKEND);
  if (e) {
    set_error("backendCreate 0x%llx", (unsigned long long)e);
    pthread_mutex_unlock(&LOCK);
    return -1;
  }
  e = IF.deviceCreate(LOG, NULL, &DEVICE);
  if (e) {
    set_error("deviceCreate 0x%llx", (unsigned long long)e);
    pthread_mutex_unlock(&LOCK);
    return -1;
  }

  /* Vote the DSP up to the max voltage corner with sleep disabled. Without an
   * explicit power config the skel stays at a low corner and low bus bandwidth,
   * which is why the big graphs (rmvpe ~190 MB of weights) run 3-5x slower than
   * the same .bin in a process that votes. ML-Sharp-QNN does the same thing. */
  QnnDevice_Infrastructure_t infra = NULL;
  if (IF.deviceGetInfrastructure && QNN_SUCCESS == IF.deviceGetInfrastructure(&infra) && infra) {
    QnnHtpDevice_Infrastructure_t *htpInfra = (QnnHtpDevice_Infrastructure_t *)infra;
    QnnHtpDevice_PerfInfrastructure_t perf = htpInfra->perfInfra;
    if (perf.createPowerConfigId && perf.setPowerConfig) {
      uint32_t cfgId = 0;
      if (QNN_HTP_PERF_INFRASTRUCTURE_ERROR_UNSUPPORTED !=
              perf.createPowerConfigId(0, 0, &cfgId) &&
          cfgId != 0) {
        QnnHtpPerfInfrastructure_PowerConfig_t pc;
        memset(&pc, 0, sizeof(pc));
        pc.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
        pc.dcvsV3Config.contextId = cfgId;
        pc.dcvsV3Config.setDcvsEnable = 1;
        pc.dcvsV3Config.dcvsEnable = 1;
        /* locked max: min=target=max pins the frequency, sleep off keeps it
         * there between graphExecute calls (this is a short burst of work) */
        pc.dcvsV3Config.setSleepDisable = 1;
        pc.dcvsV3Config.sleepDisable = 1;
        pc.dcvsV3Config.powerMode =
            QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_ADJUST_UP_DOWN;
        pc.dcvsV3Config.setBusParams = 1;
        pc.dcvsV3Config.busVoltageCornerMin =
            DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        pc.dcvsV3Config.busVoltageCornerTarget =
            DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        pc.dcvsV3Config.busVoltageCornerMax =
            DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        pc.dcvsV3Config.setCoreParams = 1;
        pc.dcvsV3Config.coreVoltageCornerMin =
            DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        pc.dcvsV3Config.coreVoltageCornerTarget =
            DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        pc.dcvsV3Config.coreVoltageCornerMax =
            DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        const QnnHtpPerfInfrastructure_PowerConfig_t *pcs[] = {&pc, NULL};
        Qnn_ErrorHandle_t pe = perf.setPowerConfig(cfgId, pcs);
        if (pe != QNN_SUCCESS) {
          qlog("perf infra: setPowerConfig 0x%llx (non-fatal)",
               (unsigned long long)pe);
          perf.destroyPowerConfigId(cfgId);
        } else {
          qlog("perf infra: locked MAX corner, sleep disabled (cfgId=%u)", cfgId);
        }
      } else {
        qlog("perf infra: createPowerConfigId unsupported");
      }
    } else {
      qlog("perf infra: createPowerConfigId/setPowerConfig unavailable");
    }
  } else {
    qlog("perf infra: deviceGetInfrastructure unavailable");
  }
#undef IF
  INIT_DONE = 1;
  qlog("qnn_init ok: backend ready");
  pthread_mutex_unlock(&LOCK);
  return 0;
}

/* Bind the graph descriptors from a system GraphInfoV1 into the slot's tensor
 * tables. Shared by the .bin and .dlc loaders — both produce the same
 * QnnSystemContext_GraphInfo_t, only the graph/context provenance differs. */
static int bind_graph(struct graph *g, const QnnSystemContext_GraphInfoV1_t *gi) {
  g->n_in = gi->numGraphInputs;
  g->n_out = gi->numGraphOutputs;
  if (g->n_in > MAX_TENSORS || g->n_out > MAX_TENSORS || g->n_in == 0 ||
      g->n_out == 0) {
    set_error("tensor count in=%u out=%u", g->n_in, g->n_out);
    return -1;
  }
  for (uint32_t i = 0; i < g->n_in; i++)
    if (describe(&gi->graphInputs[i], &g->in[i]) != 0) {
      char e[192];
      snprintf(e, sizeof(e), "%s", qnn_last_error());
      set_error("input[%u]: %s", i, e);
      return -1;
    }
  for (uint32_t i = 0; i < g->n_out; i++)
    if (describe(&gi->graphOutputs[i], &g->out[i]) != 0) {
      char e[192];
      snprintf(e, sizeof(e), "%s", qnn_last_error());
      set_error("output[%u]: %s", i, e);
      return -1;
    }
  return 0;
}

int qnn_load(const char *tag, const char *bin_path) {
  pthread_mutex_lock(&LOCK);
  if (!INIT_DONE) {
    set_error("backend not initialised");
    pthread_mutex_unlock(&LOCK);
    return -1;
  }
  struct graph *g = find_graph(tag);
  if (g) free_graph(g);
  g = NULL;
  for (int i = 0; i < MAX_GRAPHS; i++)
    if (!GRAPHS[i].used) {
      g = &GRAPHS[i];
      break;
    }
  if (!g) {
    set_error("no free graph slots");
    pthread_mutex_unlock(&LOCK);
    return -1;
  }
  memset(g, 0, sizeof(*g));
  g->fd = -1;

  const QnnSystemContext_GraphInfo_t *gis = NULL;
  uint32_t ngraphs = 0;
  const char *gname = NULL;
  struct stat st;

  g->fd = open(bin_path, O_RDONLY);
  if (g->fd < 0) {
    set_error("open %s", bin_path);
    goto fail;
  }
  if (fstat(g->fd, &st) != 0) {
    set_error("fstat %s", bin_path);
    goto fail;
  }
  g->map_len = (size_t)st.st_size;
  g->map = mmap(NULL, g->map_len, PROT_READ, MAP_PRIVATE, g->fd, 0);
  if (g->map == MAP_FAILED) {
    set_error("mmap %s", bin_path);
    goto fail;
  }

#define IF IFACE->v2_34
  if (IF.contextCreateFromBinary(BACKEND, DEVICE, NULL, g->map,
                                 (Qnn_ContextBinarySize_t)g->map_len, &g->ctx,
                                 NULL)) {
    set_error("contextCreateFromBinary failed for %s", bin_path);
    goto fail;
  }
  if (graph_infos(g->map, g->map_len, &gis, &ngraphs) != 0) {
    set_error("graph metadata for %s", bin_path);
    goto fail;
  }
  if (ngraphs != 1) {
    set_error("expected 1 graph in %s, found %u (use qnn_load_multi)",
              bin_path, ngraphs);
    goto fail;
  }
  gname = gis[0].graphInfoV1.graphName;
  if (IF.graphRetrieve(g->ctx, gname, &g->g)) {
    set_error("graphRetrieve '%s' failed", gname);
    goto fail;
  }
  if (bind_graph(g, &gis[0].graphInfoV1) != 0) {
    set_error("tensor setup for %s: %s", bin_path, qnn_last_error());
    goto fail;
  }
#undef IF

  strcpy(g->tag, tag);
  g->used = 1;
  qlog("loaded '%s' in=%u out=%u (%zu kB)", tag, g->n_in, g->n_out,
       g->map_len / 1024);
  pthread_mutex_unlock(&LOCK);
  return 0;

fail:
  free_graph(g);
  pthread_mutex_unlock(&LOCK);
  return -1;
}

/* Load every graph of a multi-graph context binary (qai-hub link job output).
 * One HTP context + one mmap is shared by all of them; each graph is registered
 * under the name stored in the binary. `names` (optional) receives up to
 * `max_names` names, `stride` bytes apart. Returns the graph count or -1. */
int qnn_load_multi(const char *bin_path, char *names, int stride, int max_names) {
  pthread_mutex_lock(&LOCK);
  if (!INIT_DONE) {
    set_error("backend not initialised");
    pthread_mutex_unlock(&LOCK);
    return -1;
  }
  int fd = -1;
  void *map = MAP_FAILED;
  size_t map_len = 0;
  struct stat st;
  struct grp *grp = NULL;
  const QnnSystemContext_GraphInfo_t *gis = NULL;
  uint32_t ngraphs = 0;
  int loaded = 0;

  for (int i = 0; i < (int)(sizeof(GRPS) / sizeof(GRPS[0])); i++)
    if (!GRPS[i].used) { grp = &GRPS[i]; break; }
  if (!grp) {
    set_error("no free context slots");
    goto fail;
  }
  memset(grp, 0, sizeof(*grp));
  grp->fd = -1;

  fd = open(bin_path, O_RDONLY);
  if (fd < 0) {
    set_error("open %s", bin_path);
    grp = NULL;
    goto fail;
  }
  if (fstat(fd, &st) != 0) {
    set_error("fstat %s", bin_path);
    grp = NULL;
    goto fail;
  }
  map_len = (size_t)st.st_size;
  map = mmap(NULL, map_len, PROT_READ, MAP_PRIVATE, fd, 0);
  if (map == MAP_FAILED) {
    set_error("mmap %s", bin_path);
    grp = NULL;
    goto fail;
  }
  if (graph_infos(map, map_len, &gis, &ngraphs) != 0) {
    set_error("graph metadata for %s: %s", bin_path, qnn_last_error());
    goto fail;
  }
  if ((int)ngraphs > max_names && names) ngraphs = (uint32_t)max_names;
  if (ngraphs > MAX_GRAPHS) ngraphs = MAX_GRAPHS;

#define IF IFACE->v2_34
  if (IF.contextCreateFromBinary(BACKEND, DEVICE, NULL, map,
                                 (Qnn_ContextBinarySize_t)map_len, &grp->ctx,
                                 NULL)) {
    set_error("contextCreateFromBinary failed for %s", bin_path);
    goto fail;
  }
  grp->fd = fd;
  grp->map = map;
  grp->map_len = map_len;
  grp->used = 1;
  fd = -1;   /* owned by the group now */
  map = MAP_FAILED;

  for (uint32_t i = 0; i < ngraphs; i++) {
    const char *gname = gis[i].graphInfoV1.graphName;
    if (!gname) continue;
    struct graph *old = find_graph(gname);
    if (old) free_graph(old);
    struct graph *g = NULL;
    for (int s = 0; s < MAX_GRAPHS; s++)
      if (!GRAPHS[s].used) { g = &GRAPHS[s]; break; }
    if (!g) {
      set_error("no free graph slots");
      goto fail;
    }
    memset(g, 0, sizeof(*g));
    g->fd = -1;
    g->grp = grp;
    grp->refs++;
    if (IF.graphRetrieve(grp->ctx, gname, &g->g)) {
      set_error("graphRetrieve '%s' failed", gname);
      g->grp = NULL;
      grp->refs--;
      goto fail;
    }
    if (bind_graph(g, &gis[i].graphInfoV1) != 0) {
      char e[192];
      snprintf(e, sizeof(e), "%s", qnn_last_error());
      set_error("multi[%u] %s (%s): %s", i, gname, bin_path, e);
      g->grp = NULL;
      grp->refs--;
      goto fail;
    }
    snprintf(g->tag, sizeof(g->tag), "%s", gname);
    g->used = 1;
    if (names && loaded < max_names) {
      snprintf(names + (size_t)loaded * stride, stride, "%s", gname);
    }
    loaded++;
    qlog("loaded '%s' (multi) in=%u out=%u", gname, g->n_in, g->n_out);
  }
#undef IF
  pthread_mutex_unlock(&LOCK);
  return loaded;

fail:
  if (fd >= 0) close(fd);
  if (map != MAP_FAILED) munmap(map, map_len);
  if (grp && grp->used) {
    if (grp->refs <= 0) {
      if (grp->ctx) IFACE->v2_34.contextFree(grp->ctx, NULL);
      munmap(grp->map, grp->map_len);
      close(grp->fd);
      memset(grp, 0, sizeof(*grp));
      grp->fd = -1;
    }
  }
  pthread_mutex_unlock(&LOCK);
  return -1;
}

/* Load a .dlc container. The DLC path differs from the .bin path in one way
 * only: the system library parses the container and composes the graphs onto a
 * fresh context itself (no contextCreateFromBinary), after which the graph is
 * retrieved by name exactly as in the .bin path. The weight memory the graphs
 * reference stays owned by the DLC handle, which is why free_graph releases it
 * only after the context is gone. */
int qnn_load_dlc(const char *tag, const char *dlc_path) {
  pthread_mutex_lock(&LOCK);
  if (!INIT_DONE) {
    set_error("backend not initialised");
    pthread_mutex_unlock(&LOCK);
    return -1;
  }
  struct graph *g = find_graph(tag);
  if (g) free_graph(g);
  g = NULL;
  for (int i = 0; i < MAX_GRAPHS; i++)
    if (!GRAPHS[i].used) {
      g = &GRAPHS[i];
      break;
    }
  if (!g) {
    set_error("no free graph slots");
    pthread_mutex_unlock(&LOCK);
    return -1;
  }
  memset(g, 0, sizeof(*g));
  g->fd = -1;

  /* declared before any goto: the fail path must not cross their init */
  void *sh = NULL;
  Qnn_ErrorHandle_t (*sysGetProviders)(const QnnSystemInterface_t ***, uint32_t *) = NULL;
  const QnnSystemInterface_t **list = NULL;
  uint32_t nsys = 0;
  const QnnSystemInterface_t *sys = NULL;
  QnnSystemContext_GraphInfo_t *graphs = NULL;
  uint32_t ngraphs = 0;
  /* declared with the others: the fail path's goto must not cross an
   * initialisation, and the finalize timing below needs these */
  Qnn_ErrorHandle_t fe = 0;
  struct timespec t0, t1;

  sh = dlopen("libQnnSystem.so", RTLD_NOW);
  if (!sh) {
    set_error("dlopen libQnnSystem.so: %s", dlerror());
    goto fail;
  }
  sysGetProviders =
      (Qnn_ErrorHandle_t(*)(const QnnSystemInterface_t ***, uint32_t *))
          dlsym(sh, "QnnSystemInterface_getProviders");
  if (!sysGetProviders) {
    set_error("dlsym QnnSystemInterface_getProviders: %s", dlerror());
    goto fail;
  }
  if (sysGetProviders(&list, &nsys)) {
    set_error("QnnSystemInterface_getProviders failed");
    goto fail;
  }
  for (uint32_t i = 0; i < nsys; i++)
    if (list[i]->systemApiVersion.major == QNN_SYSTEM_API_VERSION_MAJOR &&
        list[i]->systemApiVersion.minor >= QNN_SYSTEM_API_VERSION_MINOR) {
      sys = list[i];
      break;
    }
  if (!sys) {
    set_error("no QNN System API %u.%u provider",
              QNN_SYSTEM_API_VERSION_MAJOR, QNN_SYSTEM_API_VERSION_MINOR);
    goto fail;
  }
#define SIF sys->v1_9
#define IF  IFACE->v2_34
  if (SIF.systemDlcCreateFromFile(LOG, dlc_path, &g->dlc)) {
    set_error("systemDlcCreateFromFile failed for %s", dlc_path);
    goto fail;
  }
  if (IF.contextCreate(BACKEND, DEVICE, NULL, &g->ctx)) {
    set_error("contextCreate failed for %s", dlc_path);
    goto fail;
  }
  if (SIF.systemDlcComposeGraphs(g->dlc, NULL, 0, BACKEND, g->ctx, *IFACE,
                                 QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1,
                                 &graphs, &ngraphs)) {
    set_error("systemDlcComposeGraphs failed for %s", dlc_path);
    goto fail;
  }
  if (!graphs || ngraphs != 1) {
    set_error("expected 1 graph in %s, found %u", dlc_path, ngraphs);
    goto fail;
  }
  if (IF.graphRetrieve(g->ctx, graphs[0].graphInfoV1.graphName, &g->g)) {
    set_error("graphRetrieve '%s' failed", graphs[0].graphInfoV1.graphName);
    goto fail;
  }
  /* A composed DLC graph arrives unfinalized, unlike a context binary whose
   * graphs are ready to execute: graphExecute rejects it with
   * QNN_GRAPH_ERROR_GRAPH_NOT_FINALIZED until finalized. qnn-net-run follows
   * the same compose -> finalize -> execute sequence, so this call belongs
   * here even though QnnSystemDlc.h never mentions it. The error handle and
   * the elapsed time are logged because a full on-device HTP prepare is what
   * OOMs the phone, and its failure signature is a large error code plus a
   * long runtime. */
  clock_gettime(CLOCK_MONOTONIC, &t0);
  fe = IF.graphFinalize(g->g, NULL, NULL);
  clock_gettime(CLOCK_MONOTONIC, &t1);
  if (fe) {
    set_error("graphFinalize failed for %s: 0x%llx after %.1f s", dlc_path,
              (unsigned long long)fe,
              ((t1.tv_sec - t0.tv_sec) * 1e3 +
               (t1.tv_nsec - t0.tv_nsec) / 1e6) / 1e3);
    goto fail;
  }
  if (bind_graph(g, &graphs[0].graphInfoV1) != 0) {
    set_error("tensor setup for %s: %s", dlc_path, qnn_last_error());
    goto fail;
  }
  free(graphs);
  graphs = NULL;
#undef SIF
#undef IF

  strcpy(g->tag, tag);
  g->used = 1;
  qlog("loaded '%s' (dlc) in=%u out=%u", tag, g->n_in, g->n_out);
  pthread_mutex_unlock(&LOCK);
  return 0;

fail:
  free(graphs);
  free_graph(g);
  pthread_mutex_unlock(&LOCK);
  return -1;
}

int qnn_describe(const char *tag, QnnTensorCb in_cb, QnnTensorCb out_cb,
                 void *user) {
  pthread_mutex_lock(&LOCK);
  struct graph *g = find_graph(tag);
  if (!g) {
    set_error("no such graph '%s'", tag);
    pthread_mutex_unlock(&LOCK);
    return -1;
  }
  /* qnn_exec expects the RVC_QNN_* enum, not the raw Qnn_DataType_t, so the
   * callback contract must match it or every fp32 boundary reports int32. */
  for (uint32_t i = 0; i < g->n_in; i++)
    if (in_cb)
      in_cb(user, g->in[i].name,
            g->in[i].dtype == QNN_DATATYPE_FLOAT_32 ? RVC_QNN_FLOAT32
                                                    : RVC_QNN_INT32,
            g->in[i].rank, g->in[i].dims);
  for (uint32_t i = 0; i < g->n_out; i++)
    if (out_cb)
      out_cb(user, g->out[i].name,
             g->out[i].dtype == QNN_DATATYPE_FLOAT_32 ? RVC_QNN_FLOAT32
                                                      : RVC_QNN_INT32,
             g->out[i].rank, g->out[i].dims);
  pthread_mutex_unlock(&LOCK);
  return 0;
}

int qnn_describe_raw(const char *tag, QnnTensorCb in_cb, QnnTensorCb out_cb,
                     void *user) {
  pthread_mutex_lock(&LOCK);
  struct graph *g = find_graph(tag);
  if (!g) {
    set_error("no such graph '%s'", tag);
    pthread_mutex_unlock(&LOCK);
    return -1;
  }
  for (uint32_t i = 0; i < g->n_in; i++)
    if (in_cb)
      in_cb(user, g->in[i].name, (int)g->in[i].dtype, g->in[i].rank,
            g->in[i].dims);
  for (uint32_t i = 0; i < g->n_out; i++)
    if (out_cb)
      out_cb(user, g->out[i].name, (int)g->out[i].dtype, g->out[i].rank,
             g->out[i].dims);
  pthread_mutex_unlock(&LOCK);
  return 0;
}

int qnn_exec(const char *tag,
             const char *const *in_names, const int *in_dtypes,
             const void *const *in_data, const size_t *in_elems, size_t n_in,
             float **out_data, size_t *out_elems, size_t n_out_max,
             size_t *n_out) {
  pthread_mutex_lock(&LOCK);
  struct graph *g = find_graph(tag);
  if (!g) {
    set_error("no such graph '%s'", tag);
    pthread_mutex_unlock(&LOCK);
    return -1;
  }
  if (n_in != g->n_in) {
    set_error("'%s' expects %u inputs, got %zu", tag, g->n_in, n_in);
    pthread_mutex_unlock(&LOCK);
    return -1;
  }
  for (size_t i = 0; i < n_in; i++) {
    struct tensor *t = NULL;
    for (uint32_t j = 0; j < g->n_in; j++)
      if (!strcmp(g->in[j].name, in_names[i])) {
        t = &g->in[j];
        break;
      }
    if (!t) {
      set_error("unknown input '%s' for '%s'", in_names[i], tag);
      pthread_mutex_unlock(&LOCK);
      return -1;
    }
    const int is_f16 = (t->dtype == QNN_DATATYPE_FLOAT_16);
    int want = (t->dtype == QNN_DATATYPE_FLOAT_32 || is_f16) ? RVC_QNN_FLOAT32
                                                            : RVC_QNN_INT32;
    if (in_dtypes[i] != want) {
      set_error("input '%s' dtype mismatch (want %d got %d)", in_names[i], want,
                in_dtypes[i]);
      pthread_mutex_unlock(&LOCK);
      return -1;
    }
    const size_t esz = is_f16 ? 2 : 4;
    if (in_elems[i] * esz != t->bytes) {
      set_error("input '%s' size %zu elements != %zu bytes expected",
                in_names[i], in_elems[i], t->bytes / esz);
      pthread_mutex_unlock(&LOCK);
      return -1;
    }
    if (is_f16) {
      const float *src = (const float *)in_data[i];
      uint16_t *dst = (uint16_t *)t->buf;
      for (size_t k = 0; k < in_elems[i]; k++) dst[k] = f2h(src[k]);
    } else {
      memcpy(t->buf, in_data[i], t->bytes);
    }
  }

  Qnn_Tensor_t inT[MAX_TENSORS], outT[MAX_TENSORS];
  for (uint32_t i = 0; i < g->n_in; i++) {
    inT[i].version = QNN_TENSOR_VERSION_2;
    Qnn_TensorV2_t *t = &inT[i].v2;
    memset(t, 0, sizeof(*t));
    t->id = g->in[i].id;
    t->name = g->in[i].name;
    t->type = QNN_TENSOR_TYPE_APP_WRITE;
    t->dataFormat = g->in[i].dataFormat;
    t->dataType = g->in[i].dtype;
    t->quantizeParams = g->in[i].quantizeParams;
    t->rank = g->in[i].rank;
    t->dimensions = g->in[i].dims;
    t->memType = QNN_TENSORMEMTYPE_RAW;
    t->clientBuf.data = g->in[i].buf;
    t->clientBuf.dataSize = (uint32_t)g->in[i].bytes;
  }
  for (uint32_t i = 0; i < g->n_out; i++) {
    outT[i].version = QNN_TENSOR_VERSION_2;
    Qnn_TensorV2_t *t = &outT[i].v2;
    memset(t, 0, sizeof(*t));
    t->id = g->out[i].id;
    t->name = g->out[i].name;
    t->type = QNN_TENSOR_TYPE_APP_READ;
    t->dataFormat = g->out[i].dataFormat;
    t->dataType = g->out[i].dtype;
    t->quantizeParams = g->out[i].quantizeParams;
    t->rank = g->out[i].rank;
    t->dimensions = g->out[i].dims;
    t->memType = QNN_TENSORMEMTYPE_RAW;
    t->clientBuf.data = g->out[i].buf;
    t->clientBuf.dataSize = (uint32_t)g->out[i].bytes;
  }

#define IF IFACE->v2_34
  struct timespec ts0, ts1;
  clock_gettime(CLOCK_MONOTONIC, &ts0);
  Qnn_ErrorHandle_t err = IF.graphExecute(g->g, inT, g->n_in, outT, g->n_out,
                                          NULL, NULL);
  clock_gettime(CLOCK_MONOTONIC, &ts1);
#undef IF
  uint64_t dt_ns = (uint64_t)(ts1.tv_sec - ts0.tv_sec) * 1000000000ULL +
                   (uint64_t)ts1.tv_nsec - (uint64_t)ts0.tv_nsec;
  g->exec_n++;
  g->exec_ns_sum += dt_ns;
  if (g->exec_ns_min == 0 || dt_ns < g->exec_ns_min) g->exec_ns_min = dt_ns;
  if (dt_ns > g->exec_ns_max) g->exec_ns_max = dt_ns;
  if (err) {
    set_error("graphExecute 0x%llx on '%s'", (unsigned long long)err, tag);
    pthread_mutex_unlock(&LOCK);
    return -1;
  }

  if (g->n_out > n_out_max) {
    set_error("caller provided space for %zu outputs, graph has %u", n_out_max,
              g->n_out);
    pthread_mutex_unlock(&LOCK);
    return -1;
  }
  for (uint32_t i = 0; i < g->n_out; i++) {
    const int o_f16 = (g->out[i].dtype == QNN_DATATYPE_FLOAT_16);
    size_t elems = g->out[i].bytes / (o_f16 ? 2 : 4);
    float *dst = (float *)malloc(elems * 4);
    if (!dst) {
      for (uint32_t k = 0; k < i; k++) free(out_data[k]);
      set_error("out of memory for output %u", i);
      pthread_mutex_unlock(&LOCK);
      return -1;
    }
    if (o_f16) {
      const uint16_t *src = (const uint16_t *)g->out[i].buf;
      for (size_t k = 0; k < elems; k++) dst[k] = h2f(src[k]);
    } else if (g->out[i].dtype == QNN_DATATYPE_INT_32) {
      /* the codec graphs emit token ids as int32 (--truncate_64bit_io);
       * widen them so the caller keeps seeing float32 outputs */
      const int32_t *src = (const int32_t *)g->out[i].buf;
      for (size_t k = 0; k < elems; k++) dst[k] = (float)src[k];
    } else {
      memcpy(dst, g->out[i].buf, g->out[i].bytes);
    }
    out_data[i] = dst;
    out_elems[i] = elems;
  }
  *n_out = g->n_out;
  pthread_mutex_unlock(&LOCK);
  return 0;
}

int qnn_input_dim(const char *tag, const char *in_name, int axis) {
  pthread_mutex_lock(&LOCK);
  struct graph *g = find_graph(tag);
  if (!g) {
    pthread_mutex_unlock(&LOCK);
    return -1;
  }
  int rc = -1;
  for (uint32_t i = 0; i < g->n_in; i++)
    if (!strcmp(g->in[i].name, in_name)) {
      if (axis >= 0 && (uint32_t)axis < g->in[i].rank)
        rc = (int)g->in[i].dims[axis];
      break;
    }
  pthread_mutex_unlock(&LOCK);
  return rc;
}

int qnn_free(const char *tag) {
  pthread_mutex_lock(&LOCK);
  struct graph *g = find_graph(tag);
  if (!g) {
    set_error("no such graph '%s'", tag);
    pthread_mutex_unlock(&LOCK);
    return -1;
  }
  free_graph(g);
  qlog("freed '%s'", tag);
  pthread_mutex_unlock(&LOCK);
  return 0;
}

int qnn_graph_count(void) {
  int n = 0;
  pthread_mutex_lock(&LOCK);
  for (int i = 0; i < MAX_GRAPHS; i++)
    if (GRAPHS[i].used) n++;
  pthread_mutex_unlock(&LOCK);
  return n;
}

/* Per-graph graphExecute wall time, so a headless run can tell a DSP-bound
 * graph from a host-marshalling-bound one without a debugger attached. */
const char *qnn_timings_json(void) {
  static char buf[2048];
  pthread_mutex_lock(&LOCK);
  size_t off = 0;
  for (int i = 0; i < MAX_GRAPHS; i++) {
    struct graph *g = &GRAPHS[i];
    if (!g->used || g->exec_n == 0) continue;
    double mean_ms = (double)g->exec_ns_sum / (double)g->exec_n / 1e6;
    off += (size_t)snprintf(buf + off, sizeof(buf) - off,
                            "%s{\"tag\":\"%s\",\"n\":%llu,\"mean_ms\":%.2f,"
                            "\"min_ms\":%.2f,\"max_ms\":%.2f,\"total_ms\":%.0f}",
                            off ? "," : "", g->tag,
                            (unsigned long long)g->exec_n, mean_ms,
                            (double)g->exec_ns_min / 1e6,
                            (double)g->exec_ns_max / 1e6,
                            (double)g->exec_ns_sum / 1e6);
    if (off > sizeof(buf) - 128) break;
  }
  pthread_mutex_unlock(&LOCK);
  buf[off] = '\0';
  return buf;
}

void qnn_shutdown(void) {
  pthread_mutex_lock(&LOCK);
  for (int i = 0; i < MAX_GRAPHS; i++)
    if (GRAPHS[i].used) free_graph(&GRAPHS[i]);
  if (INIT_DONE) {
#define IF IFACE->v2_34
    if (DEVICE) IF.deviceFree(DEVICE);
    if (BACKEND) IF.backendFree(BACKEND);
    if (LOG) IF.logFree(LOG);
#undef IF
    INIT_DONE = 0;
  }
  pthread_mutex_unlock(&LOCK);
}
