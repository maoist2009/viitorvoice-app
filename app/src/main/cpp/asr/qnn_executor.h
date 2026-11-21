/* qnn_executor.h — in-process QNN HTP executor for the compiled RVC graphs.
 *
 * The rvc_npu project validated this exact QNN call chain with qnn_resident.c
 * (bit-exact against qnn-net-run on every pipeline graph). This header exposes
 * the same chain as a library so the Android app's JNI layer can drive it
 * directly instead of talking to a daemon over a pipe.
 *
 * The legacy entry points (qnn_exec) only speak FLOAT_32 / INT_32, which is
 * all the RVC graphs need. qnn_exec_raw is the dtype-generic path added for
 * the ASR graphs, whose boundaries are float16 (and int8/int16 once the
 * qai-hub quantized DLCs land).
 */
#ifndef RVC_QNN_EXECUTOR_H
#define RVC_QNN_EXECUTOR_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RVC_QNN_FLOAT32 0
#define RVC_QNN_INT32   1
#define RVC_QNN_FLOAT16 2
#define RVC_QNN_INT8    3
#define RVC_QNN_UINT8   4
#define RVC_QNN_INT16   5

/* Raw (dtype-generic) tensor descriptor for qnn_exec_raw. Quantized graphs
 * keep their scale/offset in the graph's quantize params; at the boundary the
 * only thing that matters is the storage element type, so RVC_QNN_INT8 covers
 * QNN_DATATYPE_SFIXED_POINT_8 and so on. */
typedef struct {
  const char *name;
  int        dtype;   /* RVC_QNN_* */
  void      *data;
  size_t     bytes;   /* storage bytes; must equal the tensor's byte size */
} QnnIo;

/* Errors surface as a non-zero return code; qnn_last_error() holds the text.
 * qnn_load takes a prebuilt context binary (.bin); qnn_load_dlc takes a DLC
 * container and composes its graphs onto a fresh context at load time. */
int  qnn_init(const char *lib_dir, const char *skel_dir);
int  qnn_load(const char *tag, const char *bin_path);
int  qnn_load_dlc(const char *tag, const char *dlc_path);
int  qnn_free(const char *tag);
void qnn_shutdown(void);

/* 只在**没有图**的时候拆 backend/device。每个引擎释放自己的图之后调它，
 * 这样多套模型常驻时不会互相把 QNN 上下文拆掉（qnn_shutdown 会 free 所有图）。 */
void qnn_shutdown_if_empty(void);

/* Metadata callback: one call per input / output tensor. dims has rank entries
 * and stays valid for the duration of the callback. */
typedef void (*QnnTensorCb)(void *user, const char *name, int dtype,
                            uint32_t rank, const uint32_t *dims);
int qnn_describe(const char *tag, QnnTensorCb in_cb, QnnTensorCb out_cb, void *user);

/* Execute `tag`. Each input is (name, dtype, elements); data points to
 * floats when dtype == RVC_QNN_FLOAT32 and to int32_t when INT_32. Outputs
 * come back in graph declaration order as float32; each out_data[i] is
 * malloc'd and must be free()d by the caller. */
int qnn_exec(const char *tag,
             const char *const *in_names, const int *in_dtypes,
             const void *const *in_data, const size_t *in_elems, size_t n_in,
             float **out_data, size_t *out_elems, size_t n_out_max,
             size_t *n_out);

/* dtype-generic execution. Inputs are matched by name; every buffer must be
 * exactly the tensor's byte size (see qnn_describe + qnn_elem_size). Outputs
 * are written in graph declaration order; an entry with data == NULL receives
 * a malloc'd buffer (bytes is then filled in, and the caller frees it), while
 * a non-NULL entry must already be at least the tensor's byte size. *n_out
 * reports the graph's output count. */
int qnn_exec_raw(const char *tag, const QnnIo *in, size_t n_in,
                 QnnIo *out, size_t n_out_max, size_t *n_out);

/* Bytes per element for an RVC_QNN_* dtype, or 0 if unknown. */
size_t qnn_elem_size(int dtype);

/* Number of graphs currently loaded. */
int qnn_graph_count(void);

/* Per-graph graphExecute wall-time stats as a JSON array. */
const char *qnn_timings_json(void);

const char *qnn_last_error(void);

#ifdef __cplusplus
}
#endif

#endif  /* RVC_QNN_EXECUTOR_H */
