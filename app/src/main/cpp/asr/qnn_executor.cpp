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
#include <errno.h>
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

#define TAG "rvc_native"

#define MAX_GRAPHS  16
/* The ASR decoder parts carry one past/present pair per layer (10 layers =>
 * 4 + 20 = 24 inputs, 21 outputs); 16 was tuned for the RVC graphs only. */
#define MAX_TENSORS 64
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

struct graph {
  int      used;
  char     tag[NAME_LEN];
  int      fd;
  void    *map;
  size_t   map_len;
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
    case QNN_DATATYPE_INT_16:
    case QNN_DATATYPE_UINT_16:
    case QNN_DATATYPE_SFIXED_POINT_16:
    case QNN_DATATYPE_UFIXED_POINT_16:
      return elems * 2;
    case QNN_DATATYPE_INT_8:
    case QNN_DATATYPE_UINT_8:
    case QNN_DATATYPE_BOOL_8:
    case QNN_DATATYPE_SFIXED_POINT_8:
    case QNN_DATATYPE_UFIXED_POINT_8:
      return elems;
    default:
      return 0;   /* unsupported at the boundary — refuse, never miscast */
  }
}

/* QNN dtype -> the coarse RVC_QNN_* code the JNI layer compares against.
 * Quantized storage types collapse to their width: int8-fixed and uint8-fixed
 * differ only in the zero point, which the graph applies itself. */
static int rvc_dtype_of(Qnn_DataType_t dt) {
  switch (dt) {
    case QNN_DATATYPE_FLOAT_32:
      return RVC_QNN_FLOAT32;
    case QNN_DATATYPE_FLOAT_16:
      return RVC_QNN_FLOAT16;
    case QNN_DATATYPE_INT_32:
    case QNN_DATATYPE_UINT_32:
      return RVC_QNN_INT32;
    case QNN_DATATYPE_INT_16:
    case QNN_DATATYPE_UINT_16:
    case QNN_DATATYPE_SFIXED_POINT_16:
    case QNN_DATATYPE_UFIXED_POINT_16:
      return RVC_QNN_INT16;
    case QNN_DATATYPE_INT_8:
    case QNN_DATATYPE_SFIXED_POINT_8:
      return RVC_QNN_INT8;
    case QNN_DATATYPE_UINT_8:
    case QNN_DATATYPE_BOOL_8:
    case QNN_DATATYPE_UFIXED_POINT_8:
      return RVC_QNN_UINT8;
    default:
      return -1;
  }
}

size_t qnn_elem_size(int dtype) {
  switch (dtype) {
    case RVC_QNN_FLOAT32:
    case RVC_QNN_INT32:
      return 4;
    case RVC_QNN_FLOAT16:
    case RVC_QNN_INT16:
      return 2;
    case RVC_QNN_INT8:
    case RVC_QNN_UINT8:
      return 1;
    default:
      return 0;
  }
}

static void free_graph(struct graph *g) {
#define IF IFACE->v2_34
  if (g->g) IF.graphFinalize(g->g, NULL, NULL);
  if (g->ctx) IF.contextFree(g->ctx, NULL);
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
  if (!name || strlen(name) >= NAME_LEN || rank > 8) return -1;
  strcpy(t->name, name);
  t->rank = rank;
  size_t elems = 1;
  for (uint32_t i = 0; i < rank; i++) {
    t->dims[i] = dims[i];
    elems *= dims[i];
  }
  size_t b = dtype_bytes(t->dtype, elems);
  if (!b) {
    set_error("tensor '%s' has unsupported dtype %d", name, (int)t->dtype);
    return -1;
  }
  t->bytes = b;
  void *p = NULL;
  if (posix_memalign(&p, 4096, b) != 0 || !p) return -1;
  t->buf = p;
  return 0;
}

/* 张量描述日志：load 时打印 dtype/dims/bytes，便于和云端对照。 */
static void qlog_tensor(const char *kind, uint32_t i, const struct tensor *t) {
  char d[96];
  int off = 0;
  d[0] = '\0';
  for (uint32_t k = 0; k < t->rank && off < 80; k++)
    off += snprintf(d + off, sizeof(d) - off, "%s%u", k ? "," : "", t->dims[k]);
  qlog("%s[%u] '%s' dtype=%d fmt=%d dims=[%s] bytes=%zu", kind, i, t->name,
       (int)t->dtype, (int)t->dataFormat, d, t->bytes);
}

/* libQnnSystem metadata: graph name + input/output tensor descriptors. */
static int graph_info(const void *bin, size_t len, const char **graphName,
                      const QnnSystemContext_GraphInfo_t **gi) {
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
  if (numGraphs != 1) {
    set_error("expected 1 graph, found %u", numGraphs);
    return -1;
  }
  *gi = &graphs[0];
  *graphName = graphs[0].graphInfoV1.graphName;
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
  for (uint32_t i = 0; i < g->n_in; i++) {
    if (describe(&gi->graphInputs[i], &g->in[i]) != 0) return -1;
    qlog_tensor("in", i, &g->in[i]);
  }
  for (uint32_t i = 0; i < g->n_out; i++) {
    if (describe(&gi->graphOutputs[i], &g->out[i]) != 0) return -1;
    qlog_tensor("out", i, &g->out[i]);
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

  const char *gname = NULL;
  const QnnSystemContext_GraphInfo_t *gi = NULL;

  g->fd = open(bin_path, O_RDONLY);
  if (g->fd < 0) {
    set_error("open %s failed: errno %d (%s)", bin_path, errno, strerror(errno));
    goto fail;
  }
  struct stat st;
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
  if (graph_info(g->map, g->map_len, &gname, &gi) != 0 || !gname || !gi) {
    set_error("graph metadata for %s", bin_path);
    goto fail;
  }
  if (IF.graphRetrieve(g->ctx, gname, &g->g)) {
    set_error("graphRetrieve '%s' failed", gname);
    goto fail;
  }
  if (bind_graph(g, &gi->graphInfoV1) != 0) {
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
   * callback contract must match it or every fp16 boundary reports int32. */
  for (uint32_t i = 0; i < g->n_in; i++)
    if (in_cb)
      in_cb(user, g->in[i].name, rvc_dtype_of(g->in[i].dtype), g->in[i].rank,
            g->in[i].dims);
  for (uint32_t i = 0; i < g->n_out; i++)
    if (out_cb)
      out_cb(user, g->out[i].name, rvc_dtype_of(g->out[i].dtype),
             g->out[i].rank, g->out[i].dims);
  pthread_mutex_unlock(&LOCK);
  return 0;
}

int qnn_exec_raw(const char *tag, const QnnIo *in, size_t n_in,
                 QnnIo *out, size_t n_out_max, size_t *n_out) {
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
  if (g->n_out > n_out_max) {
    set_error("caller provided space for %zu outputs, graph has %u", n_out_max,
              g->n_out);
    pthread_mutex_unlock(&LOCK);
    return -1;
  }
  for (size_t i = 0; i < n_in; i++) {
    struct tensor *t = NULL;
    for (uint32_t j = 0; j < g->n_in; j++)
      if (!strcmp(g->in[j].name, in[i].name)) {
        t = &g->in[j];
        break;
      }
    if (!t) {
      set_error("unknown input '%s' for '%s'", in[i].name, tag);
      pthread_mutex_unlock(&LOCK);
      return -1;
    }
    int want = rvc_dtype_of(t->dtype);
    if (in[i].dtype != want) {
      set_error("input '%s' dtype mismatch (want %d got %d)", in[i].name, want,
                in[i].dtype);
      pthread_mutex_unlock(&LOCK);
      return -1;
    }
    if (in[i].bytes != t->bytes) {
      set_error("input '%s' is %zu bytes, graph wants %zu", in[i].name,
                in[i].bytes, t->bytes);
      pthread_mutex_unlock(&LOCK);
      return -1;
    }
    memcpy(t->buf, in[i].data, t->bytes);
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

  for (uint32_t i = 0; i < g->n_out; i++) {
    if (!out[i].data) {
      void *p = malloc(g->out[i].bytes);
      if (!p) {
        for (uint32_t k = 0; k < i; k++)
          if (out[k].data) { free(out[k].data); out[k].data = NULL; }
        set_error("out of memory for output %u", i);
        pthread_mutex_unlock(&LOCK);
        return -1;
      }
      out[i].data = p;
      out[i].dtype = rvc_dtype_of(g->out[i].dtype);
    } else if (out[i].bytes < g->out[i].bytes) {
      set_error("output %u buffer is %zu bytes, graph writes %zu", i,
                out[i].bytes, g->out[i].bytes);
      pthread_mutex_unlock(&LOCK);
      return -1;
    }
    memcpy(out[i].data, g->out[i].buf, g->out[i].bytes);
    out[i].bytes = g->out[i].bytes;
    out[i].name = g->out[i].name;
  }
  *n_out = g->n_out;
  pthread_mutex_unlock(&LOCK);
  return 0;
}

int qnn_exec(const char *tag,
             const char *const *in_names, const int *in_dtypes,
             const void *const *in_data, const size_t *in_elems, size_t n_in,
             float **out_data, size_t *out_elems, size_t n_out_max,
             size_t *n_out) {
  if (n_in > MAX_TENSORS) {
    set_error("too many inputs (%zu)", n_in);
    return -1;
  }
  QnnIo io[MAX_TENSORS], oo[MAX_TENSORS];
  for (size_t i = 0; i < n_in; i++) {
    size_t es = qnn_elem_size(in_dtypes[i]);
    if (!es) {
      set_error("unsupported input dtype %d", in_dtypes[i]);
      return -1;
    }
    io[i].name = in_names[i];
    io[i].dtype = in_dtypes[i];
    io[i].data = (void *)in_data[i];
    io[i].bytes = in_elems[i] * es;
  }
  memset(oo, 0, sizeof(oo));
  size_t got = 0;
  if (qnn_exec_raw(tag, io, n_in, oo, n_out_max, &got)) return -1;
  for (size_t i = 0; i < got; i++) {
    out_data[i] = (float *)oo[i].data;
    out_elems[i] = oo[i].bytes / 4;   /* legacy callers are all float32 */
  }
  *n_out = got;
  return 0;
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

/* 多套模型常驻时（流式的 VAD/CAM++/SV + 对齐器），谁先跑完谁只释放自己的
 * 图；backend/device 留到最后一个图也走了再拆。 */
void qnn_shutdown_if_empty(void) {
  pthread_mutex_lock(&LOCK);
  int used = 0;
  for (int i = 0; i < MAX_GRAPHS; i++)
    if (GRAPHS[i].used) used++;
  if (used == 0 && INIT_DONE) {
#define IF IFACE->v2_34
    if (DEVICE) IF.deviceFree(DEVICE);
    if (BACKEND) IF.backendFree(BACKEND);
    if (LOG) IF.logFree(LOG);
#undef IF
    INIT_DONE = 0;
  }
  pthread_mutex_unlock(&LOCK);
}
