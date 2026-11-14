/* qnn_executor.h — in-process QNN HTP executor for the compiled RVC graphs.
 *
 * The rvc_npu project validated this exact QNN call chain with qnn_resident.c
 * (bit-exact against qnn-net-run on every pipeline graph). This header exposes
 * the same chain as a library so the Android app's JNI layer can drive it
 * directly instead of talking to a daemon over a pipe.
 *
 * All RVC graphs exchange only FLOAT_32 and INT_32 tensors at the graph
 * boundary, and every graph output is FLOAT_32.
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

/* Errors surface as a non-zero return code; qnn_last_error() holds the text.
 * qnn_load takes a prebuilt context binary (.bin); qnn_load_dlc takes a DLC
 * container and composes its graphs onto a fresh context at load time. */
int  qnn_init(const char *lib_dir, const char *skel_dir);
int  qnn_load(const char *tag, const char *bin_path);
/* Multi-graph context binary (qai-hub link job): every graph is registered
 * under the name stored in the binary, sharing one context and one mmap.
 * `names` may be NULL; otherwise up to `max_names` names are copied out,
 * `stride` bytes apart. Returns the number of graphs loaded or -1. */
int  qnn_load_multi(const char *bin_path, char *names, int stride, int max_names);
/* Dimension `axis` of input `in_name` of a loaded graph, or -1 when absent. */
int  qnn_input_dim(const char *tag, const char *in_name, int axis);
int  qnn_load_dlc(const char *tag, const char *dlc_path);
int  qnn_free(const char *tag);
void qnn_shutdown(void);

/* Metadata callback: one call per input / output tensor. dims has rank entries
 * and stays valid for the duration of the callback. */
typedef void (*QnnTensorCb)(void *user, const char *name, int dtype,
                            uint32_t rank, const uint32_t *dims);
int qnn_describe(const char *tag, QnnTensorCb in_cb, QnnTensorCb out_cb, void *user);
/* Same as qnn_describe() but hands out the raw Qnn_DataType_t. */
int qnn_describe_raw(const char *tag, QnnTensorCb in_cb, QnnTensorCb out_cb,
                     void *user);

/* Execute `tag`. Each input is (name, dtype, elements); data points to
 * floats when dtype == RVC_QNN_FLOAT32 and to int32_t when INT_32. Outputs
 * come back in graph declaration order as float32; each out_data[i] is
 * malloc'd and must be free()d by the caller. */
int qnn_exec(const char *tag,
             const char *const *in_names, const int *in_dtypes,
             const void *const *in_data, const size_t *in_elems, size_t n_in,
             float **out_data, size_t *out_elems, size_t n_out_max,
             size_t *n_out);

/* Number of graphs currently loaded. */
int qnn_graph_count(void);

/* Per-graph graphExecute wall-time stats as a JSON array. */
const char *qnn_timings_json(void);

const char *qnn_last_error(void);

#ifdef __cplusplus
}
#endif

#endif  /* RVC_QNN_EXECUTOR_H */
