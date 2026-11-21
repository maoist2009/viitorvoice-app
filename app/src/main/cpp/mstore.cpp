/* mstore.cpp — see mstore.h. */
#include "mstore.h"

#include <fcntl.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>

namespace {

struct Binding {
  char suffix[MSTORE_SUFFIX_MAX];
  int fd;
};

Binding g_bind[MSTORE_MAX];
int g_n = 0;
pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;

/* -2 = no binding matched, -1 = matched but dup failed, >= 0 = fresh fd. */
int lookup(const char *path) {
  const size_t pl = strlen(path);
  int out = -2;
  pthread_mutex_lock(&g_lock);
  for (int i = 0; i < g_n; i++) {
    const size_t sl = strlen(g_bind[i].suffix);
    if (sl <= pl && memcmp(path + pl - sl, g_bind[i].suffix, sl) == 0) {
      out = dup(g_bind[i].fd);
      break;
    }
  }
  pthread_mutex_unlock(&g_lock);
  return out;
}

}  // namespace

extern "C" int mstore_bind(const char *path_suffix, int fd) {
  if (!path_suffix || !*path_suffix || fd < 0) return -1;
  const size_t sl = strlen(path_suffix);
  if (sl >= MSTORE_SUFFIX_MAX) return -1;
  pthread_mutex_lock(&g_lock);
  for (int i = 0; i < g_n; i++) {
    if (strcmp(g_bind[i].suffix, path_suffix) == 0) {
      close(g_bind[i].fd);
      g_bind[i].fd = fd;
      pthread_mutex_unlock(&g_lock);
      return 0;
    }
  }
  if (g_n >= MSTORE_MAX) {
    pthread_mutex_unlock(&g_lock);
    return -1;
  }
  memcpy(g_bind[g_n].suffix, path_suffix, sl + 1);
  g_bind[g_n].fd = fd;
  g_n++;
  pthread_mutex_unlock(&g_lock);
  return 0;
}

extern "C" int mstore_bound(void) {
  pthread_mutex_lock(&g_lock);
  int n = g_n;
  pthread_mutex_unlock(&g_lock);
  return n;
}

extern "C" void mstore_clear(void) {
  pthread_mutex_lock(&g_lock);
  for (int i = 0; i < g_n; i++) close(g_bind[i].fd);
  g_n = 0;
  pthread_mutex_unlock(&g_lock);
}

extern "C" int mstore_open(const char *path) {
  if (!path) return -1;
  const int fd = lookup(path);
  if (fd != -2) return fd;
  return open(path, O_RDONLY);
}

extern "C" FILE *mstore_fopen(const char *path) {
  if (!path) return nullptr;
  const int fd = lookup(path);
  if (fd == -2) return fopen(path, "rb");
  if (fd < 0) return nullptr;
  FILE *f = fdopen(fd, "rb");
  if (!f) close(fd);
  return f;
}
