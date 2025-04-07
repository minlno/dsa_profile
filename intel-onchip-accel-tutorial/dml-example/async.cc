#include <dml/dml.hpp>
#include <iostream>
#include <string>
#include <chrono>
#include "common.h"



int async(uint8_t *data_buf[][BUF_SIZE]) {
  int status = 1;
  uint64_t start;
  uint64_t submit = 0;
  uint64_t wait = 0;

  dml::handler<dml::mem_move_operation, std::allocator<dml::byte_t>> handlers[BUF_SIZE];
  dml::mem_move_result results[BUF_SIZE];



  /////////////////////////////////////////////////////////////////////////////
  // NOTE: Could replace both submit and wait lines with comment below:
  // auto result = dml::execute<path>(dml::mem_move, dml::make_view(data_buf[0][i]), dml::make_view(data_buf[1][i]));
  // Descriptor Submission
  for (int i = 0; i < BUF_SIZE; i++) {

    start = rdtsc();
    handlers[i] = dml::submit<dml::hardware>(dml::mem_move, dml::data_view(data_buf[0][i], XFER_SIZE/sizeof(uint8_t)),
                                                             dml::data_view(data_buf[1][i], XFER_SIZE/sizeof(uint8_t)));
    submit += rdtsc() - start;
  }
  /////////////////////////////////////////////////////////////////////////////



  /////////////////////////////////////////////////////////////////////////////
  // Wait for Completion
  for (int i = 0; i < BUF_SIZE; i++) {
    start = rdtsc();
    results[i] = handlers[i].get();
    wait += rdtsc() - start;
  }
  /////////////////////////////////////////////////////////////////////////////



  for (int i = 0; i < BUF_SIZE; i++) {
    if (status == 1)
      status = results[i].status == dml::status_code::ok;
  }

  // Print times
  std::cout << "[time  ] submission: " << std::dec << submit << std::endl;
  std::cout << "[time  ] waiting: " << std::dec << wait << std::endl;
  std::cout << "[time  ] full offload: " << std::dec << submit + wait << std::endl;

  return status;
}

