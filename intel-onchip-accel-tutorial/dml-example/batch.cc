#include <dml/dml.hpp>
#include <iostream>
#include <string>
#include <chrono>
#include "common.h"



int batch(uint8_t *data_buf[][BUF_SIZE]) {
  int status = 1;
  uint64_t start;
  uint64_t prepare = 0;
  uint64_t submit = 0;
  uint64_t wait = 0;

  auto sequence = dml::sequence(BATCH_SIZE, std::allocator<dml::byte_t>());



  /////////////////////////////////////////////////////////////////////////////
  // Descriptor Preparation
  for (int i = 0; i < BATCH_SIZE; i++) {
    start = rdtsc();
    sequence.add(dml::mem_move, dml::data_view(data_buf[0][i], XFER_SIZE/sizeof(uint8_t)),
                                dml::data_view(data_buf[1][i], XFER_SIZE/sizeof(uint8_t)));
    prepare += rdtsc() - start;
  }
  /////////////////////////////////////////////////////////////////////////////



  /////////////////////////////////////////////////////////////////////////////
  // Descriptor Submission
  // NOTE: Could replace both submit and wait lines with comment below:
  // auto result = dml::execute<path>(dml::mem_move, dml::make_view(data_buf[0][i]), dml::make_view(data_buf[1][i]));
  start = rdtsc();
  auto handler = dml::submit<dml::hardware>(dml::batch, sequence);
  submit += rdtsc() - start;
  /////////////////////////////////////////////////////////////////////////////



  /////////////////////////////////////////////////////////////////////////////
  // Wait for Completion
  start = rdtsc();
  auto result = handler.get();
  wait += rdtsc() - start;
  /////////////////////////////////////////////////////////////////////////////



  if (status == 1)
    status = result.status == dml::status_code::ok;

  // Print times
  std::cout << "[time  ] preparation: " << std::dec << prepare << std::endl;
  std::cout << "[time  ] submission: " << std::dec << submit << std::endl;
  std::cout << "[time  ] wait: " << std::dec << wait << std::endl;
  std::cout << "[time  ] full offload: " << std::dec << prepare + submit + wait << std::endl;

  return status;
}

