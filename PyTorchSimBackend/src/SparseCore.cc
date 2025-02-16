#include "SparseCore.h"

SparseCore::SparseCore(uint32_t id, SimulationConfig config) : Core(id, config) {
  stonneCore = new SST_STONNE::sstStonne(config.stonne_config_path);
  stonneCore->init(1);
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
  std::cout << "Pending Requests: " << _request_queue.size() << std::endl;
  std::cout << "Pending Responses: " << _response_queue.size() << std::endl;
}

void SparseCore::print_current_stats() {
  std::cout << "Current SparseCore Status:" << std::endl;
  print_stats();
}