#include "DMA.h"
#include "TileGraph.h"

DMA::DMA(uint32_t id, uint32_t dram_req_size) {
  _id = id;
  _dram_req_size = dram_req_size;
  _current_inst = nullptr;
  _finished = true;
}

void DMA::issue_tile(std::shared_ptr<Instruction> inst) {
  _current_inst = std::move(inst);
  std::vector<size_t>& tile_size = _current_inst->get_tile_size();
  if (tile_size.size() <= 0 || tile_size.size() > get_max_dim()) {
    spdlog::error("[DMA {}] issued tile is not supported format..", _id);
    exit(EXIT_FAILURE);
  }
  _finished = false;
}

std::shared_ptr<std::vector<mem_fetch*>> DMA::get_memory_access(cycle_type core_cycle) {
  auto addr_set = _current_inst->get_dram_address(_dram_req_size);
  auto access_vec = std::make_shared<std::vector<mem_fetch *>>();
  Tile* owner = (Tile*)_current_inst->get_owner();
  std::shared_ptr<TileSubGraph> owner_subgraph = owner->get_owner();
  unsigned long long base_daddr = _current_inst->get_base_dram_address();
  // Todo. We use a ternsor level buffer allocation, so we don't need to check all memfetch
  bool is_cacheable = owner_subgraph->is_cacheable(base_daddr, base_daddr + _dram_req_size);
  spdlog::trace("[{}][Core {}][SRAM] Address: 0x{:016x}, Is_cacheable: {}", core_cycle, _id, base_daddr, is_cacheable);
  spdlog::trace("[{}][Core {}][NUMA] Subgraph id: {} , Numa id: {}, Arg: {} is_write: {}",
    core_cycle, _id, owner_subgraph->get_core_id(), _current_inst->get_numa_id(), _current_inst->get_addr_name(), _current_inst->is_dma_write());

  for (auto addr: *addr_set) {
    mem_access_type acc_type = _current_inst->is_dma_write() ? mem_access_type::GLOBAL_ACC_W : mem_access_type::GLOBAL_ACC_R;
    mf_type type = _current_inst->is_dma_write() ? mf_type::WRITE_REQUEST : mf_type::READ_REQUEST;
    mem_fetch* access = new mem_fetch(addr, acc_type, type, _dram_req_size, _current_inst->get_numa_id(), static_cast<void*>(_current_inst.get()));
    access->set_cacheable(is_cacheable);
    _current_inst->inc_waiting_request();
    access_vec->push_back(access);
  }
  _finished = true;
  return access_vec;
}

uint32_t DMA::generate_mem_access_id() {
  static uint32_t id_counter{0};
  return id_counter++;
}