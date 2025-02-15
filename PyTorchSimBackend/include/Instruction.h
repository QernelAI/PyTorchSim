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

enum class Opcode { MOVIN, MOVOUT, COMP, BAR, COUNT};

typedef uint64_t addr_type;
typedef uint64_t cycle_type;

std::string opcode_to_string(Opcode opcode);

class Instruction {
 public:
  Instruction(Opcode opcode, cycle_type compute_cycle, size_t num_parents, addr_type dram_addr,
              std::vector<size_t> tile_size, size_t precision, std::vector<int> &idx_list,
              std::vector<int> &stride_list,  std::vector<int> tag_idx_list, std::vector<int> tag_stride_list,
              std::vector<int> accum_tag_idx_list, std::vector<int> loop_size_list);
  void finish_instruction();
  void add_child(std::shared_ptr<Instruction> child);
  bool check_ready() { return ready_counter == 0; }
  const Opcode get_opcode() { return opcode; }
  bool is_dma_read() { return opcode == Opcode::MOVIN; }
  bool is_dma_write() { return opcode == Opcode::MOVOUT; }
  bool is_async_dma() { return _is_async_dma; }
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
  void set_compute_cycle(cycle_type cycle) { compute_cycle = cycle; }
  void print();
  std::set<addr_type> get_dram_address(addr_type dram_req_size) {
    std::set<addr_type> address_set;

    /* Set 4D shape*/
    while (tile_size.size() < 4)
      tile_size.insert(tile_size.begin(), 1);

    while (_stride_list.size() < 4)
      _stride_list.insert(_stride_list.begin(), 1);

    /* Iterate tile_size */
    for (int dim0=0; dim0<tile_size.at(0); dim0++) {
      for (int dim1=0; dim1<tile_size.at(1); dim1++) {
        for (int dim2=0; dim2<tile_size.at(2); dim2++) {
          for (int dim3=0; dim3<tile_size.at(3); dim3++) {
            addr_type address = dim0*_stride_list.at(_stride_list.size() - 4) + \
                                dim1*_stride_list.at(_stride_list.size() - 3) + \
                                dim2*_stride_list.at(_stride_list.size() - 2) + \
                                dim3*_stride_list.at(_stride_list.size() - 1);
            address = dram_addr + address * _precision;
            address_set.insert(address - (address & dram_req_size-1));
          }
        }
      }
    }
    return address_set;
  }
  size_t get_free_sram_size() { return _free_sram_size; }
  void adjust_dram_address() {
    int offset = std::inner_product(_idx_list.begin(), _idx_list.end(), _stride_list.begin(), 0);
    dram_addr += offset * _precision;
  }
  void set_free_sram_size(size_t sram_size) { _free_sram_size=sram_size; }
  void* get_owner() { return _owner; }
  void set_owner(void *owner) { _owner = owner;}
  void set_compute_type(int type) { _compute_type = type; }
  int get_compute_type() { return _compute_type; }
  void set_numa_id(int numa_id) { _numa_id = numa_id; }
  uint32_t get_numa_id() { return _numa_id; }
  std::vector<int>& get_idx_list() { return _idx_list; }
  std::vector<int>& get_tag_idx_list() { return _tag_idx_list; }
  std::vector<int>& get_tag_stride_list() { return _tag_stride_list; }
  std::vector<int>& get_tag_id() { return _tag_key; }
  void set_addr_name(std::string name) { _addr_name = name; }
  std::string get_addr_name() { return _addr_name; }
  void set_nr_inner_loop(int nr) { _nr_inner_loop = nr; }
  int get_nr_inner_loop() { return _nr_inner_loop; }
  void set_is_async(bool is_async) { _is_async_dma = is_async; }

  cycle_type start_cycle;
  cycle_type finish_cycle;
  cycle_type bubble_cycle=0;

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
  size_t _tile_numel;
  size_t _nr_waiting_request=0;
  size_t _precision=0;
  size_t _free_sram_size=0;
  addr_type dram_addr;
  uint32_t _numa_id = 0; // For DMA instruction
  int _compute_type = 0;
  std::vector<int> _idx_list;
  std::vector<int> _stride_list;
  std::vector<int> _tag_idx_list;
  std::vector<int> _tag_stride_list;
  std::vector<int> _tag_key;
  std::vector<int> _accum_tag_idx_list;
  std::vector<int> _loop_size_list;
  std::string _addr_name;
  int _nr_inner_loop = 0;
  bool _is_async_dma=false;
};