#include <stdio.h>
#include <stdint.h>
#include "common.h"
#include <linux/idxd.h>



int single(uint64_t *(data_buf[][BUF_SIZE]), struct dsa_hw_desc *desc_buf,
                                             struct dsa_completion_record *comp_buf,
                                             void *wq_portal, int opcode,
					     struct dsa_delta_record *dr_buf[]) {
  int retry, status;
  uint64_t start;
  uint64_t prep = 0;
  uint64_t submit = 0;
  uint64_t wait = 0;


  // Submit 4 seprate single offloads in a row
  for (int i = 0; i < BUF_SIZE; i++) {

    ///////////////////////////////////////////////////////////////////////////
    // Descriptor Preparation
    start = rdtsc();

    comp_buf[i].status          = 0;
    desc_buf[i].flags           = IDXD_OP_FLAG_RCR | IDXD_OP_FLAG_CRAV;
    desc_buf[i].xfer_size       = XFER_SIZE;

    if (opcode == DSA_OPCODE_MEMMOVE) {
    	desc_buf[i].opcode          = DSA_OPCODE_MEMMOVE;
    	desc_buf[i].src_addr        = (uintptr_t)data_buf[0][i];
    	desc_buf[i].dst_addr        = (uintptr_t)data_buf[1][i];
    	desc_buf[i].completion_addr = (uintptr_t)&(comp_buf[i]);
    } else if (opcode == DSA_OPCODE_CR_DELTA) {
    	desc_buf[i].opcode          = DSA_OPCODE_CR_DELTA;
	printf("opcode: %x\n", desc_buf[i].opcode);
	printf("xfer_size: %d\n", desc_buf[i].xfer_size);
    	desc_buf[i].src_addr        = (uintptr_t)data_buf[0][i];
    	desc_buf[i].src2_addr        = (uintptr_t)data_buf[1][i];
    	desc_buf[i].completion_addr = (uintptr_t)&(comp_buf[i]);
	desc_buf[i].delta_addr = (uint64_t)(uintptr_t)dr_buf[i];
	//desc_buf[i].max_delta_size = XFER_SIZE / 8 * 10;
	desc_buf[i].max_delta_size = XFER_SIZE * 10; // only for 8B offload test
    }

    prep += rdtsc() - start;
    ///////////////////////////////////////////////////////////////////////////



    ///////////////////////////////////////////////////////////////////////////
    // Descriptor Submission
    start = rdtsc();

    _mm_sfence();
    /* movdir64b(wq_portal, &desc_buf[i]); */
    enqcmd(wq_portal, &desc_buf[i]);

    submit += rdtsc() - start;
    ///////////////////////////////////////////////////////////////////////////



    ///////////////////////////////////////////////////////////////////////////
    // Wait for Completion
    retry = 0;
    start = rdtsc();

    while (comp_buf[i].status == 0 && retry++ < MAX_COMP_RETRY) {
      umonitor(&(comp_buf[i]));
      if (comp_buf[i].status == 0) {
        uint64_t delay = __rdtsc() + UMWAIT_DELAY;
        umwait(UMWAIT_STATE_C0_1, delay);
      }
    }

    wait += rdtsc() - start;
    ///////////////////////////////////////////////////////////////////////////
  }



  // Print times
  printf("[time  ] preparation: %lu\n", prep);
  printf("[time  ] submission: %lu\n", submit);
  printf("[time  ] wait: %lu\n", wait);
  printf("[time  ] full offload: %lu\n", prep + submit + wait);
  printf("[avg time  ] preparation: %lu\n", prep / BUF_SIZE);
  printf("[avg time  ] submission: %lu\n", submit / BUF_SIZE);
  printf("[avg time  ] wait: %lu\n", wait / BUF_SIZE);
  printf("[avg time  ] full offload: %lu\n", (prep + submit + wait) / BUF_SIZE);

  status = 1;
  for (int i = 0; i < BUF_SIZE; i++) {
    if (comp_buf[i].status != 1) {
      status = comp_buf[i].status;
      break;
    }
  }

  return status;
}
