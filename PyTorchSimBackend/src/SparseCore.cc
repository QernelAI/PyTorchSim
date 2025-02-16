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

  double compute_throughput = static_cast<double>(num_ms) * core_freq / 1e3; // FLOPs/sec
  double dn_bandwidth = static_cast<double>(dn_bw) * dn_width * core_freq * 1e6 / 8.0 / 1e9; // GB/s
  double rn_bandwidth = static_cast<double>(rn_bw) * rn_width * core_freq * 1e6 / 8.0 / 1e9; // GB/s

  spdlog::info("[Config/StonneCore {}] Compute Throughput: {:.2f} GFLOPs/sec", id, compute_throughput);
  spdlog::info("[Config/StonneCore {}] Distribution Network Bandwidth: {:.2f} GB/s ({} ports x {} bits)",
             id, dn_bandwidth, dn_bw, dn_width);
  spdlog::info("[Config/StonneCore {}] Reduction Network Bandwidth: {:.2f} GB/s ({} ports x {} bits)",
             id, rn_bandwidth, rn_bw, rn_width);
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
  stonneCore->cycle();

  /* Send Memory Request */
  while (SimpleMem::Request* req = stonneCore->popRequest()) {
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
    mem_fetch* req_wrapper = new mem_fetch(req->getAddress(), acc_type, type, _config.dram_req_size, -1, req);
    _request_queue.push(req_wrapper);
  }

  /* Send Memory Response */
  while (!_response_queue.empty()) {
    mem_fetch* resp_wrapper = _response_queue.front();
    SimpleMem::Request* resp = static_cast<SimpleMem::Request*>(resp_wrapper->get_custom_data());
    resp->setReply();
    stonneCore->pushResponse(resp);
    _response_queue.pop();
    delete resp_wrapper;
  }

  if (stonneCore->isFinished()) {
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