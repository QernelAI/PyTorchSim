#pragma once

#include <robin_hood.h>
#include <spdlog/fmt/ranges.h>
#include <spdlog/spdlog.h>
#include <numeric>

#include <set>
#include <cassert>
#include <cstdint>
#include <memory>
#include <vector>

enum class Opcode { MOVIN, MOVOUT, COMP, BAR};

typedef uint64_t addr_type;
typedef uint64_t cycle_type;

std::string opcode_to_string(Opcode opcode);

class Instruction {
 public:
  Instruction(Opcode opcode, cycle_type compute_cycle, size_t num_parents, addr_type dram_addr,
              std::vector<size_t> tile_size, std::vector<size_t> tile_stride, size_t precision,
              std::vector<int> &idx_list, std::vector<int> tag_idx_list, std::vector<int> loop_size_list);
  void finish_instruction();
  void add_child(std::shared_ptr<Instruction> child);
  bool check_ready() { return ready_counter == 0; }
  const Opcode get_opcode() { return opcode; }
  bool is_dma_read() { return opcode == Opcode::MOVIN; }
  bool is_dma_write() { return opcode == Opcode::MOVOUT; }
  bool is_async_dma() { return _tag_idx_list.size() >= 2; } // FIXME.
  bool is_ready() { return ready_counter == 0; }
  void inc_ready_counter() { ready_counter++; }
  void dec_ready_counter() {
    assert(ready_counter!=0);
    ready_counter--;
  }
  size_t get_tile_numel() { return _tile_numel; }
  size_t get_precision() { return _precision; }
  void inc_waiting_request();
  void dec_waiting_request();
  size_t get_waiting_request() { return _nr_waiting_request; }
  std::vector<size_t>& get_tile_size() { return tile_size; }
  void set_overlapping_cycle(cycle_type cycle) { overlapping_cycle = cycle; }
  cycle_type get_overlapping_cycle() { return overlapping_cycle; }
  cycle_type get_compute_cycle() { return compute_cycle; }
  void print();
  // lamda function to get the dram address
  addr_type get_dram_address(int row, int col) {
    auto get_tile_address = [this](size_t i, size_t j) -> addr_type {
      return dram_addr + (i * tile_stride[0] + j) * _precision;
    };
    if (_idx_list.size() >= 2) {
      int len = _idx_list.size();
      return get_tile_address(_idx_list.at(len-2) + row, _idx_list.at(len-1) + col);
    } else if (_idx_list.size() == 1) {
      return get_tile_address(row, _idx_list.at(0) + col);
    } else {
      spdlog::error("Calculating DRAM address error...");
      exit(EXIT_FAILURE);
    }
  }
  size_t get_free_sram_size() { return _free_sram_size; }
  void adjust_dram_address() {
    int start_pos = _idx_list.size() - _nr_inner_loop * 2;
    int end_pos = _idx_list.size() - _nr_inner_loop;

    std::vector<int> new_idx(_idx_list.begin(), _idx_list.begin() + end_pos);
    std::vector<int> sizes(_loop_size_list.begin(), _loop_size_list.begin() + end_pos);
    for (int i=0; i < _nr_inner_loop; i++) {
      new_idx.at(start_pos+i) += _idx_list.at(end_pos + i);
    }

    std::vector<int> strides(new_idx.size(), 1);
    for (int i = sizes.size() - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * sizes[i + 1];
    }
    int offset = std::inner_product(strides.begin(), strides.end(), new_idx.begin(), 0);
    dram_addr += offset;
  }
  void set_free_sram_size(size_t sram_size) { _free_sram_size=sram_size; }
  void* get_owner() { return _owner; }
  void set_owner(void *owner) { _owner = owner;}
  void set_compute_type(int type) { _compute_type = type; }
  int get_compute_type() { return _compute_type; }
  std::vector<int>& get_idx_list() { return _idx_list; }
  std::vector<int>& get_tag_idx_list() { return _tag_idx_list; }
  void set_addr_name(std::string name) { _addr_name = name; }
  std::string get_addr_name() { return _addr_name; }
  void set_nr_inner_loop(int nr) { _nr_inner_loop = nr; }
  int get_nr_inner_loop() { return _nr_inner_loop; }

  cycle_type start_cycle;
  cycle_type finish_cycle;

  bool finished=false;
  int subgraph_id;
 private:
  void *_owner;
  Opcode opcode;
  cycle_type compute_cycle;
  cycle_type overlapping_cycle;
  size_t ready_counter;
  std::set<std::shared_ptr<Instruction>> child_inst;
  std::vector<size_t> tile_size;
  std::vector<size_t> tile_stride;
  size_t _tile_numel;
  size_t _nr_waiting_request=0;
  size_t _precision=0;
  size_t _free_sram_size=0;
  addr_type dram_addr;
  int _compute_type = 0;
  std::vector<int> _idx_list;
  std::vector<int> _tag_idx_list;
  std::vector<int> _loop_size_list;
  std::string _addr_name;
  int _nr_inner_loop = 0;
};