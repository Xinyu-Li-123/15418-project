#pragma once

#define PROJ_DEBUG
// #undef PROJ_DEBUG

#include <cerrno>
#include <ctime>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>

#ifdef PROJ_DEBUG
#define LogInfo(fmt, ...)                                                      \
  do {                                                                         \
    fprintf(stdout, "[Info] [%s:%d %s] " fmt "\n", __FILE__, __LINE__,         \
            __func__, ##__VA_ARGS__);                                          \
    fflush(stdout);                                                            \
  } while (0)
#else
#define LogInfo(...)                                                           \
  do {                                                                         \
    (void)0;                                                                   \
  } while (0)
#endif

#ifdef PROJ_DEBUG
#define ProcLogInfo(fmt, ...)                                                  \
  do {                                                                         \
    fprintf(stdout, "(pid=%d) [Info] [%s:%d %s] " fmt "\n", pid, __FILE__,     \
            __LINE__, __func__, ##__VA_ARGS__);                                \
    fflush(stdout);                                                            \
  } while (0)
#else
#define ProcLogInfo(...)                                                       \
  do {                                                                         \
    (void)0;                                                                   \
  } while (0)
#endif

#ifdef PROJ_DEBUG
#define Assert(cond, fmt, ...)                                                 \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "[Error] Assertion failed: %s:%d: %s: (%s) " fmt "\n",   \
              __FILE__, __LINE__, __func__, #cond, ##__VA_ARGS__);             \
      fflush(stderr);                                                          \
      abort();                                                                 \
    }                                                                          \
  } while (0)
#else
#define Assert(...)                                                            \
  do {                                                                         \
    (void)0;                                                                   \
  } while (0)
#endif

#define Abort(...) Assert(0, __VA_ARGS__)
