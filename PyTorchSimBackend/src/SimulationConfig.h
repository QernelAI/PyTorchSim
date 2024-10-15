#pragma once

#include <nlohmann/json.hpp>
#include <string>

using json = nlohmann::json;

enum class DramType { SIMPLE, RAMULATOR1, RAMULATOR2 };

enum class IcntType { SIMPLE, BOOKSIM2 };

struct SimulationConfig {
  /* Core config */
  uint32_t num_cores;
  uint32_t core_freq;
  uint32_t sram_size;
  uint32_t core_print_interval = 0;

  /* DRAM config */
  DramType dram_type;
  uint32_t dram_freq;
  uint32_t dram_channels;
  uint32_t dram_req_size;
  uint32_t dram_latency;
  uint32_t dram_print_interval;
  std::string dram_config_path;

  /* ICNT config */
  IcntType icnt_type;
  std::string icnt_config_path;
  uint32_t icnt_freq;
  uint32_t icnt_latency;
  uint32_t icnt_print_interval=0;

  /* Sheduler config */
  uint32_t num_patition=1;
  std::string scheduler_type;

  /* Core id, Partiton id mapping */
  std::map<uint32_t, uint32_t> partiton_map;

  /* Other configs */
  uint32_t precision;
  std::string layout;

  uint64_t align_address(uint64_t addr) {
    return addr - (addr % dram_req_size);
  }

  float max_dram_bandwidth() {
    return dram_freq * dram_channels * dram_req_size / 1000; // GB/s
  }
};