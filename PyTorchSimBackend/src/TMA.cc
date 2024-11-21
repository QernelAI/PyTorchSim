#include "TMA.h"
#include "TileGraph.h"

TMA::TMA(uint32_t id, uint32_t dram_req_size) {
  _id = id;
  _dram_req_size = dram_req_size;
  _current_inst = nullptr;
  _finished = true;
}

void TMA::issue_tile(std::shared_ptr<Instruction> inst) {
  _current_inst = std::move(inst);
  std::vector<size_t>& tile_size = _current_inst->get_tile_size();
  if (tile_size.size() != 2) {
    spdlog::error("[TMA {}] issued tile is not [y,x] format..", _id);
    exit(EXIT_FAILURE);
  }
  _finished = false;
}

std::vector<MemoryAccess*> TMA::get_memory_access() {
  std::set<addr_type> addr_set = _current_inst->get_dram_address(_dram_req_size);
  std::vector<MemoryAccess *> access_vec;
  Tile* owner = (Tile*)_current_inst->get_owner();
  TileSubGraph* owner_subgraph = owner->get_owner();
  spdlog::trace("[NUMA Trace] Subgraph id: {} , Numa id: {}, Arg: {} is_write: {}",
    owner_subgraph->get_core_id(), _current_inst->get_numa_id(), _current_inst->get_addr_name(), _current_inst->is_dma_write());
  for (auto addr: addr_set) {
    MemoryAccess* access = new MemoryAccess({
      .id = generate_mem_access_id(),
      .dram_address = addr,
      .size = _dram_req_size,
      .write = _current_inst->is_dma_write(),
      .request = true,
      .numa_id = _current_inst->get_numa_id(),
      .owner_instruction = _current_inst.get()
    });
    _current_inst->inc_waiting_request();
    access_vec.push_back(access);
  }
  _finished = true;
  return access_vec;
}

uint32_t TMA::generate_mem_access_id() {
  static uint32_t id_counter{0};
  return id_counter++;
}