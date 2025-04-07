#ifndef _COMMON_H__
#define __COMMON_H__

#include <dml/dml.hpp>
#include <x86intrin.h>

#ifndef __GNUC__
#define __asm__ asm
#endif

#define EXAMPLES      3
#define XFER_SIZE  4096
#define BATCH_SIZE    4
#define BUF_SIZE      4

int single(uint8_t *data_buf[][BUF_SIZE]);
int batch(uint8_t *data_buf[][BUF_SIZE]);
int async(uint8_t *data_buf[][BUF_SIZE]);

static __always_inline uint64_t
rdtsc(void) {
  uint64_t tsc;
  unsigned int dummy;
  tsc = __rdtscp(&dummy);
  __builtin_ia32_lfence();
  return tsc;
}

static inline
void clflushopt(volatile void *__p) {
  asm volatile("clflushopt %0" : "+m" (*(volatile char  *)__p));
}

static inline void
cflush(char *buf, uint64_t len) {
  char *b = buf;
  char *e = buf + len;
  for (; b < e; b += 64)
    clflushopt(b);
}

#endif
