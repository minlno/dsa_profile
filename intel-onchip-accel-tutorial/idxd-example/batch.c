#include <stdio.h>
#include <stdint.h>
#include "common.h"
#include <linux/idxd.h>



int batch(uint64_t *(data_buf[][BUF_SIZE]), struct dsa_hw_desc *desc_buf,
                                            struct dsa_completion_record *comp_buf,
                                            void *wq_portal) {
  int retry;
  uint64_t start, offload_start;
  uint64_t prep = 0;
  uint64_t submit = 0;
  uint64_t wait = 0;



  /////////////////////////////////////////////////////////////////////////////
  // Descriptor Preparation
  start = offload_start = rdtsc();

  for (int i = 0; i < BATCH_SIZE; i++) {
    comp_buf[i].status          = 0;
    desc_buf[i].opcode          = DSA_OPCODE_MEMMOVE;
    desc_buf[i].flags           = IDXD_OP_FLAG_RCR | IDXD_OP_FLAG_CRAV;
    desc_buf[i].xfer_size       = XFER_SIZE;
    desc_buf[i].src_addr        = (uintptr_t)data_buf[0][i];
    desc_buf[i].dst_addr        = (uintptr_t)data_buf[1][i];
    desc_buf[i].completion_addr = (uintptr_t)&(comp_buf[i]);
  }

  // Prepare batch descriptor
  struct dsa_completion_record batch_comp __attribute__((aligned(32)));
  struct dsa_hw_desc batch_desc = { };
  batch_comp.status          = 0;
  batch_desc.opcode          = DSA_OPCODE_BATCH;
  batch_desc.flags           = IDXD_OP_FLAG_RCR | IDXD_OP_FLAG_CRAV;
  batch_desc.desc_count      = BATCH_SIZE;
  batch_desc.desc_list_addr  = (uint64_t)&(desc_buf[0]);
  batch_desc.completion_addr = (uintptr_t)&batch_comp;

  prep = rdtsc() - start;
  /////////////////////////////////////////////////////////////////////////////



  /////////////////////////////////////////////////////////////////////////////
  // Descriptor Submission
  start = rdtsc();

  _mm_sfence();
  /* movdir64b(wq_portal, &batch_desc); */
  enqcmd(wq_portal, &batch_desc);

  submit = rdtsc() - start;
  /////////////////////////////////////////////////////////////////////////////



  /////////////////////////////////////////////////////////////////////////////
  // Wait for Completion
  retry = 0;
  start = rdtsc();

  while (batch_comp.status == 0 && retry++ < MAX_COMP_RETRY) {
    umonitor(&batch_comp);
    if (batch_comp.status == 0) {
      uint64_t delay = __rdtsc() + UMWAIT_DELAY;
      umwait(UMWAIT_STATE_C0_1, delay);
    }
  }

  wait = rdtsc() - start;
  /////////////////////////////////////////////////////////////////////////////



  // Print times
  printf("[time  ] preparation: %lu\n", prep);
  printf("[time  ] submission: %lu\n", submit);
  printf("[time  ] wait: %lu\n", wait);
  printf("[time  ] full offload: %lu\n", prep + submit + wait);

  return batch_comp.status;
}
