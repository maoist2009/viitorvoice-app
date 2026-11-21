/* mstore.h — model blobs addressed through virtual paths.
 *
 * The shared model library (MLP/1) never hands out host paths: a
 * ContentProvider passes read-only fds into the app. This module lets the
 * pipelines keep their plain-path code: bind a suffix once ("/llm.bin") and
 * every later open of a path that ends with that suffix resolves to a dup()
 * of the bound fd. Unbound paths fall through to the filesystem, so an app
 * with private copies keeps working unchanged.
 */
#ifndef VV_MSTORE_H
#define VV_MSTORE_H

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MSTORE_MAX 64
#define MSTORE_SUFFIX_MAX 160

/* Register `fd` (read-only, ownership transferred) for `path_suffix`.
 * Returns 0 on success, -1 when the table is full or the arguments are bad. */
int  mstore_bind(const char *path_suffix, int fd);

/* Number of live bindings. */
int  mstore_bound(void);

/* Drop every binding and close the borrowed fds. Contexts already mapped from
 * them stay valid: mmap keeps the file alive on its own. */
void mstore_clear(void);

/* open(path, O_RDONLY) that honours the bindings. Caller owns the fd. */
int  mstore_open(const char *path);

/* fopen(path, "rb") that honours the bindings. Caller fcloses. */
FILE *mstore_fopen(const char *path);

#ifdef __cplusplus
}
#endif

#endif /* VV_MSTORE_H */
