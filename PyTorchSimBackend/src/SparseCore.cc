#include "SparseCore.h"

SparseCore::SparseCore(uint32_t id, SimulationConfig config) : Core(id, config) {
  stonneCore = new SST_STONNE::sstStonne(config.stonne_config_path);
  stonneCore->init(1);
  Config stonneConfig = stonneCore->getStonneConfig();
  unsigned int core_freq = config.core_freq; // MHz;
  unsigned int num_ms = stonneConfig.m_MSNetworkCfg.ms_size;
  unsigned int dn_bw = stonneConfig.m_SDMemoryCfg.n_read_ports;
  unsigned int dn_width = stonneConfig.m_SDMemoryCfg.port_width;
  unsigned int rn_bw = stonneConfig.m_SDMemoryCfg.n_write_ports;
  unsigned int rn_width = stonneConfig.m_SDMemoryCfg.port_width;
  r_port_nr = dn_bw;
  w_port_nr = rn_bw;

  double compute_throughput = static_cast<double>(num_ms) * core_freq / 1e3; // FLOPs/sec
  double dn_bandwidth = static_cast<double>(dn_bw) * dn_width * core_freq * 1e6 / 8.0 / 1e9; // GB/s
  double rn_bandwidth = static_cast<double>(rn_bw) * rn_width * core_freq * 1e6 / 8.0 / 1e9; // GB/s

  spdlog::info("[Config/StonneCore {}] Compute Throughput: {:.2f} GFLOPs/sec", id, compute_throughput);
  spdlog::info("[Config/StonneCore {}] Distribution Network Bandwidth: {:.2f} GB/s ({} ports x {} bits)",
             id, dn_bandwidth, r_port_nr, dn_width);
  spdlog::info("[Config/StonneCore {}] Reduction Network Bandwidth: {:.2f} GB/s ({} ports x {} bits)",
             id, rn_bandwidth, w_port_nr, rn_width);
};

SparseCore::~SparseCore() { delete stonneCore; }

bool SparseCore::running() {
  return !_request_queue.empty() || !_response_queue.empty() || _tiles.size();
}

void SparseCore::issue(std::shared_ptr<Tile> tile) {
  SST_STONNE::StonneOpDesc *opDesc = static_cast<SST_STONNE::StonneOpDesc*>(tile->get_custom_data());
  stonneCore->setup(*opDesc);
  stonneCore->init(1);
  _tiles.push_back(tile);
};

bool SparseCore::can_issue(const std::shared_ptr<Tile>& op) {
  return !running() && op->is_stonne_tile();
}

void SparseCore::cycle() {
  _core_cycle++;
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
    std::tuple<uint64_t, mem_access_type, mf_type> key = std::make_tuple(target_addr, acc_type, type);
    if (request_merge_table.find(key) == request_merge_table.end())
      request_merge_table[key] = new std::vector<SimpleMem::Request*> ();
    request_merge_table[key]->push_back(req);
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

      spdlog::debug("[SparseCore][{}] Address: {:#x}, Access Type: {}, Request Type: {}, DRAM Req Size: {}, nr_request: {}", \
              _core_cycle, req_wrapper->get_addr(), int(req_wrapper->get_access_type()), int(req_wrapper->get_type()), _config.dram_req_size, nr_request);
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
    stonneCore->pushResponse(resp);
    resps->erase(resps->begin());
    if (resps->empty()) {
      delete resps;
      delete resp_wrapper;
      _response_queue.pop();
    }
    if (nr_request++ > w_port_nr)
      break;
  }

  if (stonneCore->isFinished() && _tiles.size()) {
    stonneCore->finish();
    std::shared_ptr<Tile> target_tile = _tiles.front();
    target_tile->set_status(Tile::Status::FINISH);
    _finished_tiles.push(target_tile);
    _tiles.erase(_tiles.begin());
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
  stonneCore->printStats();
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