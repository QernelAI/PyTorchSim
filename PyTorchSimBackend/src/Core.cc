#include "Core.h"

Core::Core(uint32_t id, SimulationConfig config)
    : _id(id),
      _config(config),
      _core_cycle(0),
      _stat_tma_cycle(0),
      _tma(id, config.dram_req_size) {
  _sram_size = _config.sram_size * 1024;
  _used_sram_size = 0;
  _compute_pipeline.resize(NR_COMPUTE_UNIT);
}

bool Core::can_issue(const std::shared_ptr<Tile>& op) {
  /* Check SRAM is enough to run tile */
  return op->get_required_sram_size() + _used_sram_size <= _sram_size && _tiles.size() < 2;
}

void Core::issue(std::shared_ptr<Tile> op) {
  if (op->get_instructions().size()){
    spdlog::trace("[Core {}][{}] New Tile is issued, remain sram: {} Required size: {}, Free size: {}",
      _id, _core_cycle, _sram_size-_used_sram_size, op->get_required_sram_size(), op->get_instructions().back()->get_free_sram_size());
  } else {
    spdlog::trace("[Core {}][{}] New Tile is issued, remain sram: {} Required size: {}",
      _id, _core_cycle, _sram_size-_used_sram_size, op->get_required_sram_size());
  }
  _used_sram_size += op->get_required_sram_size();
  _tiles.push_back(std::move(op));
}

std::shared_ptr<Tile> Core::pop_finished_tile() {
  std::shared_ptr<Tile> result = std::make_unique<Tile>(Tile(Tile::Status::EMPTY));
  if (_finished_tiles.size() > 0) {
    result = std::move(_finished_tiles.front());
    _finished_tiles.pop();
  }
  return result;
}

void Core::compute_cycle() {
  for (int i=0; i<NR_COMPUTE_UNIT; i++) {
    auto& target_pipeline = _compute_pipeline.at(i);
    if (!target_pipeline.empty()) {
      _stat_compute_cycle[i]++;
      if(target_pipeline.front()->finish_cycle <= _core_cycle) {
        finish_instruction(target_pipeline.front());
        target_pipeline.pop();
      }
    } else
      _stat_compute_idle_cycle[i]++;
  }
}

void Core::dma_cycle() {
  /* Check finished dma operation */
  for (int i=0; i<_dma_waiting_queue.size(); i++){
    std::shared_ptr<Instruction>& instruction = _dma_waiting_queue.at(i);
    /* Pass not finished instruction */
    if (instruction->get_waiting_request())
      continue;

    /* Finish DMA read instruction */
    if (instruction->is_dma_read() && !instruction->is_async_dma())
      finish_instruction(instruction);

    /* Set tag table of async dma load */
    if (instruction->is_dma_read() && instruction->is_async_dma()) {
      std::ostringstream oss;
      auto key = std::make_pair(instruction->get_addr_name(), instruction->get_tag_idx_list());
      assert(!_tma.get_tag_finish(instruction->subgraph_id, key));
      _tma.set_tag_finish(instruction->subgraph_id, key);
      for (const auto& idx : instruction->get_tag_idx_list())
        oss << idx << ", ";
      spdlog::trace("[Core {}][{}] {} ASYNC FINISHED, Used sram: {}, Release sram: {}, subgraph_id: {} addr_name: {} tag_idx_list: {}",
                    _id, _core_cycle, opcode_to_string(instruction->get_opcode()),
                    _used_sram_size, instruction->get_free_sram_size(),
                    instruction->subgraph_id, instruction->get_addr_name(), oss.str());
      for (auto & wait_inst : _tma.get_tag_waiter(instruction->subgraph_id, key)) {
        finish_instruction(wait_inst);
      }
    }

    /* Erase the instruction in DMA waiting queue */
    _dma_waiting_queue.erase(_dma_waiting_queue.begin() + i);
    i--;
  }

  if (_tma.is_finished()) {
    /* Finish instruction when it is DMA store */
    if (_tma.get_current_inst() != nullptr) {
      std::shared_ptr<Instruction> finished_inst = std::move(_tma.get_current_inst());
      if (finished_inst->is_dma_write()) {
        /* Only DMA write operation is finished! */
        finish_instruction(finished_inst);
      } else if (finished_inst->is_dma_read() && finished_inst->is_async_dma()) {
        /* Register tag table for async dma load */
        _tma.register_tag(finished_inst->subgraph_id,
                          std::make_pair(finished_inst->get_addr_name(), finished_inst->get_tag_idx_list()));
        finish_instruction(finished_inst);
      } else if(!finished_inst->is_dma_read()) {
        spdlog::error("[Core {}][{}] TMA instruction in not valid", _id, _core_cycle);
        exit(EXIT_FAILURE);
      } else if (finished_inst->get_opcode() == Opcode::BAR) {
        std::ostringstream oss;
        for (const auto& idx : finished_inst->get_tag_idx_list())
          oss << idx << ", ";
        spdlog::trace("[Core {}][{}] {} FINISHED, addr_name: {} tag_list: {}", _id, _core_cycle,
                      opcode_to_string(finished_inst->get_opcode()), finished_inst->get_addr_name(), oss.str());
      }
      /*Pass to waiting queue */
      _dma_waiting_queue.push_back(std::move(finished_inst));
    }

    /* Issue new DMA operation */
    if (!_ld_inst_queue.empty()) {
      std::shared_ptr<Instruction> inst = _ld_inst_queue.front();
      _tma.issue_tile(inst);
      _ld_inst_queue.pop();
    } else if (!_st_inst_queue.empty()) {
      std::shared_ptr<Instruction> inst = _st_inst_queue.front();
      _tma.issue_tile(inst);
      _st_inst_queue.pop();
    } else {
      /* TMA is idle */
      _stat_tma_idle_cycle++;
      return;
    }
  }
  /* Generate MemoryAccess */
  while (1) {
    MemoryAccess *access = _tma.get_memory_access();
    if (access == nullptr)
      return;

    //spdlog::debug("[TMA {}] access: 0x{:x}, write: {}", _id, access->dram_address, access->write);
    /* Access couldn't be nullptr, since it is not finished */
    assert(access != nullptr);

    access->core_id = _id;
    access->start_cycle = _core_cycle;
    _request_queue.push(access);
  }

  /* Increase tma stat cycle */
  _stat_tma_cycle++;
}

void Core::cycle() {
  /* Run compute unit and DMA unit */
  compute_cycle();
  dma_cycle();

  /* Increase core cycle counter */
  _core_cycle++;

  /* Iterate tile while an instruction is issued */
  bool issued = false;

  for (int i=0; i<_tiles.size() && !issued; i++) {
    auto& instructions = _tiles[i]->get_instructions();
    for (int j=0; j<instructions.size(); j++) {
      auto& inst = instructions.at(j);
      /* Skip instruction is not ready  */
      if (!inst->is_ready())
        continue;

      switch (inst->get_opcode()) {
        case Opcode::MOVIN:
          {
            std::ostringstream oss;
            for (const auto& idx : inst->get_tag_idx_list())
              oss << idx << ", ";
            spdlog::trace("[Core {}][{}] {} ISSUED, free_sram_size: {} addr_name: {} tag_idx_list: {}", _id, _core_cycle,
                          opcode_to_string(inst->get_opcode()), inst->get_free_sram_size(),
                          inst->get_addr_name(), oss.str());
            _ld_inst_queue.push(inst);
            issued = true;
          }
          break;
        case Opcode::MOVOUT:
          spdlog::trace("[Core {}][{}] {} ISSUED, free_sram_size: {}", _id, _core_cycle,
                        opcode_to_string(inst->get_opcode()), inst->get_free_sram_size());
          _st_inst_queue.push(inst);
          issued = true;
          break;
        case Opcode::COMP:
          {
            auto& target_pipeline = _compute_pipeline.at(inst->get_compute_type());
            if (target_pipeline.empty())
              inst->finish_cycle = _core_cycle + inst->get_compute_cycle();
            else {
              int overlapped_cycle = std::min(target_pipeline.back()->finish_cycle - _core_cycle, inst->get_overlapping_cycle());
              inst->finish_cycle = target_pipeline.back()->finish_cycle + inst->get_compute_cycle() - overlapped_cycle;
            }
            spdlog::trace("[Core {}][{}] {}-{} ISSUED, finsh at {}", _id, _core_cycle,
                          opcode_to_string(inst->get_opcode()), inst->get_compute_type(), inst->finish_cycle);
            target_pipeline.push(inst);
            issued = true;
          }
          break;
        case Opcode::BAR:
          {
            std::ostringstream oss;
            auto key = std::make_pair(inst->get_addr_name(), inst->get_tag_idx_list());
            bool finished = _tma.get_tag_finish(inst->subgraph_id, key);
            if (finished) {
              finish_instruction(inst);
            } else {
              _tma.register_tag_waiter(inst->subgraph_id, key, inst);
            }
            issued = true;
          }
          break;
        default:
          spdlog::error("Undefined instruction opcode type");
          exit(EXIT_FAILURE);
      }

      if (issued) {
        auto it = instructions.begin() + j; // Position 2 is the third element
        instructions.erase(it);
        break;
      }
    }
  }

  /* Remove finshed tiles */
  bool retry = true;
  while (retry) {
    for (int i=0; i<_tiles.size() && !issued; i++) {
      if (_tiles[i]->all_insts_finshed()) {
        _tiles[i]->set_status(Tile::Status::FINISH);
        _finished_tiles.push(std::move(_tiles[i]));
        _tiles.erase(_tiles.begin() + i); // FIXME. Inefficient data structure
        /* Let's retry */
        break;
      }
    }
    retry = false;
  }
  if(_config.core_print_interval && _core_cycle % _config.core_print_interval == 0) {
    print_current_stats();
  }
}

void Core::finish_instruction(std::shared_ptr<Instruction>& inst) {
  size_t free_sram_size = inst->get_free_sram_size();
  if (inst->finished) {
    spdlog::error("[Core {}][{}] {} FINISHED, inst already finished!!", _id, _core_cycle,
                  opcode_to_string(inst->get_opcode()));
    exit(EXIT_FAILURE);
  }
  inst->finish_instruction();
  static_cast<Tile*>(inst->get_owner())->inc_finished_inst();
  if (inst->get_opcode() == Opcode::COMP) {
    spdlog::trace("[Core {}][{}] {}-{} FINISHED, Used sram: {}, Release sram: {}",
      _id, _core_cycle, opcode_to_string(inst->get_opcode()), inst->get_compute_type(),
      _used_sram_size, inst->get_free_sram_size());
  } else if (inst->get_opcode() != Opcode::BAR && inst->is_async_dma()){
    std::ostringstream oss;
    for (const auto& idx : inst->get_tag_idx_list())
      oss << idx << ", ";
    spdlog::trace("[Core {}][{}] {} ASYNC REGISTERED, Used sram: {}, Release sram: {} subgraph_id: {} addr_name: {} tag_idx_list: {}",
      _id, _core_cycle, opcode_to_string(inst->get_opcode()), _used_sram_size,
      inst->get_free_sram_size(), inst->subgraph_id, inst->get_addr_name(), oss.str());
  } else if ((inst->get_opcode() == Opcode::MOVIN || inst->get_opcode() == Opcode::MOVOUT) && !inst->is_async_dma()) {
    spdlog::trace("[Core {}][{}] {} FINISHED, free_sram_size: {} addr_name: {}", _id, _core_cycle,
      opcode_to_string(inst->get_opcode()), inst->get_free_sram_size(),
      inst->get_addr_name());
  }
  //_used_sram_size -= free_sram_size;
}

bool Core::running() {
  bool running = false;
  running = running || _tiles.size() > 0;
  for (int i=0; i<NR_COMPUTE_UNIT;i++)
    running = running || !_compute_pipeline.at(i).empty();
  running = running || !_dma_waiting_queue.empty();
  running = running || !_tma.empty();
  running = running || !_ld_inst_queue.empty();
  running = running || !_st_inst_queue.empty();
  return running;
}

bool Core::has_memory_request() { return _request_queue.size() > 0; }

void Core::pop_memory_request() {
  _request_queue.pop();
}

void Core::push_memory_response(MemoryAccess *response) {
  Instruction * owner_inst = response->owner_instruction;

  assert(owner_inst);
  assert(owner_inst->get_waiting_request());

  owner_inst->dec_waiting_request();
  delete response;
}

bool Core::can_issue_compute(std::shared_ptr<Instruction>& inst) {
  return inst->is_ready();
}

void Core::print_stats() {
  update_stats();
  spdlog::info(
      "Core [{}] : MatMul active cycle {} Vector active cycle {} ",
      _id, _stat_tot_compute_cycle[SYSTOLIC_ARRAY], _stat_tot_compute_cycle[VECTOR_UNIT]);
  spdlog::info(
      "Core [{}] : TMA active cycle {} TMA idle cycle {} Systolic Array idle cycle {} Vector unit idle cycle {}",
      _id, _stat_tot_tma_cycle, _stat_tot_tma_idle_cycle, _stat_tot_compute_idle_cycle[SYSTOLIC_ARRAY], _stat_compute_idle_cycle[VECTOR_UNIT]);
  spdlog::info("Core [{}] : Systolic Array Utilization(%) {:.2f}, Vector Unit Utilization(%) {:.2f}, Total cycle: {}",
    _id, static_cast<float>(_stat_tot_compute_cycle[SYSTOLIC_ARRAY] * 100) / _core_cycle,
    static_cast<float>(_stat_tot_compute_cycle[VECTOR_UNIT] * 100) / _core_cycle, _core_cycle);
}

void Core::print_current_stats() {
  auto level = spdlog::level::info;
  if(_id != 0)
    level = spdlog::level::debug;
  spdlog::log(level,
      "Core [{}] : MatMul active cycle {} Vector active cycle {} ",
      _id, _stat_compute_cycle[SYSTOLIC_ARRAY], _stat_compute_cycle[VECTOR_UNIT]);
  spdlog::log(level,
      "Core [{}] : TMA active cycle {} TMA idle cycle {} Systolic Array idle cycle {} Vector unit idle cycle {}",
      _id, _stat_tma_cycle, _stat_tma_idle_cycle, _stat_compute_idle_cycle[SYSTOLIC_ARRAY], _stat_compute_idle_cycle[VECTOR_UNIT]);
  spdlog::log(level,
      "Core [{}] : Systolic Array Utilization(%) {:.2f}, Vector Unit Utilization(%) {:.2f}, Total cycle: {}",
      _id, static_cast<float>(_stat_compute_cycle[SYSTOLIC_ARRAY] * 100) / _config.core_print_interval,
      static_cast<float>(_stat_compute_cycle[VECTOR_UNIT] * 100) / _config.core_print_interval, _core_cycle);
  update_stats();
}

void Core::update_stats() {
  _stat_tot_compute_cycle[SYSTOLIC_ARRAY] += _stat_compute_cycle[SYSTOLIC_ARRAY];
  _stat_tot_compute_cycle[VECTOR_UNIT] += _stat_compute_cycle[VECTOR_UNIT];
  _stat_tot_tma_cycle += _stat_tma_cycle;
  _stat_tot_tma_idle_cycle += _stat_tma_idle_cycle;
  _stat_tot_compute_idle_cycle[SYSTOLIC_ARRAY] += _stat_compute_idle_cycle[SYSTOLIC_ARRAY];
  _stat_compute_idle_cycle[VECTOR_UNIT] += _stat_compute_idle_cycle[VECTOR_UNIT];

  _stat_compute_cycle[SYSTOLIC_ARRAY] = 0;
  _stat_compute_cycle[VECTOR_UNIT] = 0;
  _stat_tma_cycle = 0;
  _stat_tma_idle_cycle = 0;
  _stat_compute_idle_cycle[SYSTOLIC_ARRAY] = 0;
  _stat_compute_idle_cycle[VECTOR_UNIT] = 0;
}