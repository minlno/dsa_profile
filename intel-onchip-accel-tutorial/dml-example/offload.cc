#include <dml/dml.hpp>
#include <random>
#include <iostream>
#include "common.h"


int main(int argc, char **argv)
{
  uint8_t* data_buf[2][BUF_SIZE];
  std::random_device rd;
  std::mt19937 rng(rd());
  std::uniform_int_distribution<uint64_t> dist(0, 65536);
  int status = 1;

  if (argc > 2) {
    std::cout << "[ERROR] Incorrect format: sudo ./dml_offload <[single, batch, or async]>" << std::endl;
    return 1;
  }

  // Allocate data buffer
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < BUF_SIZE; j++)
      data_buf[i][j] = new uint8_t[XFER_SIZE];

  // Run EXAMPLES number of offload examples
  for (int example; example < EXAMPLES; example++) {
    if (example != 0)
      std::cout << std::endl << std::endl << std::endl;
    std::cout << "             Example-" << example << std::endl;
    std::cout << "======================================" << std::endl;

    // Set random values to buffers and flush them from cache
    for (int j = 0; j < BUF_SIZE; j++) {
      for (int i = 0; i < 2; i++)
        data_buf[i][j][0] = dist(rng);
      std::cout << "[data  ] before-" << j << ": src=" << std::hex << (int)data_buf[0][j][0] <<
                   ", dst=" << std::hex << (int)data_buf[1][j][0] << std::endl;
      cflush((char*)data_buf[0][j], XFER_SIZE);
      cflush((char*)data_buf[1][j], XFER_SIZE);
    }
    std::cout << "--------------------------------------" << std::endl;



    ///////////////////////////////////////////////////////////////////////////
    // Begin offloading
    std::string offload_type = (argc == 1) ? "single" : argv[1];
    if (argc == 1 || offload_type == "single") {
      std::cout << "[info  ] running " << offload_type << " offload" << std::endl;
      status = single(data_buf);
    } else if (offload_type == "batch") {
      std::cout << "[info  ] running " << offload_type << " offload" << std::endl;
      status = batch(data_buf);
    } else if (offload_type == "async") {
      std::cout << "[info  ] running " << offload_type << " offload" << std::endl;
      status = async(data_buf);
    } else {
      std::cout << "Unknown offload type: " << offload_type << std::endl;
      return 1;
    }
    ///////////////////////////////////////////////////////////////////////////



    // Verify results
    std::cout << "--------------------------------------" << std::endl;
    if (status != 1)
      std::cout << "[verify] failed: " << std::hex << status << std::endl;
    else
      std::cout << "[verify] passed" << std::endl;

    for (int j = 0; j < BUF_SIZE; j++)
      std::cout << "[data  ] after-" << j << ": src=" << std::hex << (int)data_buf[0][j][0] <<
                   ", dst=" << std::hex << (int)data_buf[1][j][0] << std::endl;
  }

  // Cleanup
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < BUF_SIZE; j++)
      delete[] data_buf[i][j];

  return 0;
}
