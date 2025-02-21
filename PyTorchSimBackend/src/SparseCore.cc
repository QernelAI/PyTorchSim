#include "SparseCore.h"

SparseCore::SparseCore(uint32_t id, SimulationConfig config) : Core(id, config) {
  /* Init stonne cores*/
  nr_cores = config.num_stonne_per_core;
  coreBusy.resize(nr_cores);
  percore_tiles.resize(nr_cores);
  stonneCores.resize(nr_cores);
  for (int i=0; i<nr_cores; i++) {
    SST_STONNE::sstStonne* core = new SST_STONNE::sstStonne(config.stonne_config_path);
    stonneCores.at(i) = core;
    stonneCores.at(i)->init(1);
    coreBusy.at(i) = false;
    percore_tiles.at(i) = std::vector<std::shared_ptr<Tile>>();
  }

  Config stonneConfig = stonneCores.at(0)->getStonneConfig();
  unsigned int core_freq = config.core_freq; // MHz;
  unsigned int num_ms = stonneConfig.m_MSNetworkCfg.ms_size;
  r_port_nr = config.num_stonne_port;
  w_port_nr = config.num_stonne_port;

  double compute_throughput = static_cast<double>(num_ms) * core_freq / 1e3; // FLOPs/sec
  double dn_bandwidth = static_cast<double>(r_port_nr) * config.dram_req_size * core_freq * 1e6 / 8.0 / 1e9; // GB/s
  double rn_bandwidth = static_cast<double>(w_port_nr) * config.dram_req_size * core_freq * 1e6 / 8.0 / 1e9; // GB/s
  for (int i=0; i<nr_cores; i++) {
    spdlog::info("[Config/StonneCore {}][{}] Compute Throughput: {:.2f} GFLOPs/sec", id, i, compute_throughput);
    spdlog::info("[Config/StonneCore {}][{}] Distribution Network Bandwidth: {:.2f} GB/s",
                id, i, dn_bandwidth, r_port_nr);
    spdlog::info("[Config/StonneCore {}][{}] Reduction Network Bandwidth: {:.2f} GB/s",
                id, i, rn_bandwidth, w_port_nr);
  }
};

SparseCore::~SparseCore() {
  for (auto& core : stonneCores){
    delete core;
  }
}

bool SparseCore::running() {
  bool is_running = !_request_queue.empty() || !_response_queue.empty();
  for (auto& tile_vec : percore_tiles)
    is_running |= tile_vec.size();
  return is_running;
}

void SparseCore::issue(std::shared_ptr<Tile> tile) {
  int32_t selected_core_idx = -1;
  for (int i=0; i<nr_cores; i++) {
    int32_t core_idx = rr_idx % nr_cores;
    if (!coreBusy.at(i)) {
      selected_core_idx = core_idx;
      rr_idx = (selected_core_idx + 1) % nr_cores;
      break;
    }
  }
  if (selected_core_idx == -1) {
    spdlog::error("[StonneCore {}] Faield to issue tile", _id);
    exit(1);
  }
  //delete stonneCores.at(selected_core_idx);
  //SST_STONNE::sstStonne* core = new SST_STONNE::sstStonne(_config.stonne_config_path);
  //stonneCores.at(selected_core_idx) = core;
  stonneCores.at(selected_core_idx)->init(1);

  spdlog::info("[StonneCore {}][{}] issued new tile", _id, selected_core_idx);
  SST_STONNE::StonneOpDesc *opDesc = static_cast<SST_STONNE::StonneOpDesc*>(tile->get_custom_data());
  stonneCores.at(selected_core_idx)->setup(*opDesc, 0x1000000 * selected_core_idx); // FIXME. To avoid same address
  stonneCores.at(selected_core_idx)->init(1);
  percore_tiles.at(selected_core_idx).push_back(tile);
  coreBusy.at(selected_core_idx) = true;
};

bool SparseCore::can_issue(const std::shared_ptr<Tile>& op) {
  bool idle_exist = false;
  for (bool flag : coreBusy) {
    idle_exist |= !flag;
  }
  return idle_exist && op->is_stonne_tile();
}

void SparseCore::cycle() {
  _core_cycle++;
  uint32_t stonne_core_id = 0;
  for (auto& stonneCore : stonneCores) {
    stonneCore->cycle();

    /* Send Memory Request */
    while (SimpleMem::Request* req = stonneCore->popRequest()) {
      uint64_t target_addr =  (req->getAddress() / _config.dram_req_size) * _config.dram_req_size;
      mem_access_type acc_type;
      mf_type type;

      switch(req->getcmd()) {
        case SimpleMem::Request::Read:
          acc_type = mem_access_type::GLOBAL_ACC_R;
          type = mf_type::READ_REQUEST;
          break;
        case SimpleMem::Request::Write:
          acc_type = mem_access_type::GLOBAL_ACC_W;
          type = mf_type::WRITE_REQUEST;
          break;
        default:
          spdlog::error("[SparseCore] Invalid request type from core");
          return;
      }
      req->request_time = _core_cycle;
      req->stonneId = stonne_core_id;
      std::tuple<uint64_t, mem_access_type, mf_type> key = std::make_tuple(target_addr, acc_type, type);
      if (request_merge_table.find(key) == request_merge_table.end())
        request_merge_table[key] = new std::vector<SimpleMem::Request*> ();
      request_merge_table[key]->push_back(req);
    }

    if (coreBusy.at(stonne_core_id) && stonneCore->isFinished()) {
      stonneCore->finish();

      std::shared_ptr<Tile> target_tile = percore_tiles.at(stonne_core_id).front();
      target_tile->set_status(Tile::Status::FINISH);
      _finished_tiles.push(target_tile);
      percore_tiles.at(stonne_core_id).erase(percore_tiles.at(stonne_core_id).begin());
      coreBusy.at(stonne_core_id) = false;
    }
    stonne_core_id++;
  }

  int nr_request = 0;
  while (!request_merge_table.empty() && nr_request <= r_port_nr) {
    for (auto& req_pair : request_merge_table) {
      uint64_t address;
      mem_access_type acc_type;
      mf_type type;
      std::tie(address, acc_type, type) = req_pair.first;
      mem_fetch* req_wrapper = new mem_fetch(address, acc_type, type, _config.dram_req_size, -1, req_pair.second);
      _request_queue.push(req_wrapper);
      request_merge_table.erase(req_pair.first);

      spdlog::debug("[SparseCore][{}][{}] Address: {:#x}, Access Type: {}, Request Type: {}, DRAM Req Size: {}, nr_request: {}", \
              _core_cycle, stonne_core_id, req_wrapper->get_addr(), int(req_wrapper->get_access_type()), int(req_wrapper->get_type()), _config.dram_req_size, nr_request);
      nr_request++;
      break;
    }
  }

  // Send Memory Response
  nr_request = 0;
  while (!_response_queue.empty()) {
    mem_fetch* resp_wrapper = _response_queue.front();
    std::vector<SimpleMem::Request*>* resps = static_cast<std::vector<SimpleMem::Request*>*>(resp_wrapper->get_custom_data());

    SimpleMem::Request* resp = resps->front();

    spdlog::debug("[SparseCore][{}] Round Trip Cycle: {}, Address: {:#x}, Access Type: {}, Request Type: {}, DRAM Req Size: {}, nr_request: {}", \
            _core_cycle, _core_cycle - resp->request_time, resp->getAddress(), int(resp_wrapper->get_access_type()), int(resp_wrapper->get_type()), _config.dram_req_size, nr_request);

    resp->setReply();
    stonneCores.at(resp->stonneId)->pushResponse(resp);
    resps->erase(resps->begin());
    if (resps->empty()) {
      delete resps;
      delete resp_wrapper;
      _response_queue.pop();
    }
    if (nr_request++ > w_port_nr)
      break;
  }
  if(_config.core_print_interval && _core_cycle % _config.core_print_interval == 0) {
    print_current_stats();
  }
}

bool SparseCore::has_memory_request() {
  return !_request_queue.empty();
}

void SparseCore::pop_memory_request() {
  _request_queue.pop();
}

void SparseCore::push_memory_response(mem_fetch* response) {
  _response_queue.push(response);
}

void SparseCore::print_stats() {
  //for (auto stonneCore : stonneCores)
  //  stonneCore->printStats();
  MSwitchStats accum;
  spdlog::info("========= Sparse Core stat =========");
  spdlog::info("Stonne Core [{}] : Total cycle {}", _id, _core_cycle);
  for (size_t i = 0; i < stonneCores.size(); ++i) {
    MSwitchStats stats = stonneCores.at(i)->getMSStats();
    accum += stats;
    spdlog::info("Stonne Core [{}][{}] : n_multiplications: {} ",
                 _id, i, stats.n_multiplications);
  }
  spdlog::info("Stonne Core [{}] : total_multiplications: {} ",
                 _id, accum.n_multiplications);
}

void SparseCore::print_current_stats() {
  print_stats();
}

std::shared_ptr<Tile> SparseCore::pop_finished_tile() {
  std::shared_ptr<Tile> result = std::make_unique<Tile>(Tile(Tile::Status::EMPTY));
  if (_finished_tiles.size() > 0) {
    result = std::move(_finished_tiles.front());
    _finished_tiles.pop();
  }
  return result;
}