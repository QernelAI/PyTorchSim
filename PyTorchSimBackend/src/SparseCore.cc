#include "SparseCore.h"

SparseCore::SparseCore(uint32_t id, SimulationConfig config) : Core(id, config) {
  stonneCore = new SST_STONNE::sstStonne(config.stonne_config_path);

  // Dummy instruction
  SST_STONNE::StonneOpDesc opDesc;
  opDesc.operation = Layer_t::outerProductGEMM;
  opDesc.GEMM_K = 512;
  opDesc.GEMM_N = 64;
  opDesc.GEMM_M = 64;
  opDesc.GEMM_T_K = 4;
  opDesc.GEMM_T_N = 1;
  opDesc.mem_init = "/workspace/sstStonne/tests/outerproduct/outerproduct_gemm_mem.ini";
  opDesc.mem_matrix_c_file_name = "/workspace/sstStonne/tests/outerproduct/result.out";
  opDesc.matrix_a_dram_address = 0;
  opDesc.matrix_b_dram_address = 12444;
  opDesc.matrix_c_dram_address = 24608;
  opDesc.rowpointer_matrix_a_init = "/workspace/sstStonne/tests/outerproduct/outerproduct_gemm_rowpointerA.in";
  opDesc.colpointer_matrix_a_init = "/workspace/sstStonne/tests/outerproduct/outerproduct_gemm_colpointerA.in";
  opDesc.rowpointer_matrix_b_init = "/workspace/sstStonne/tests/outerproduct/outerproduct_gemm_rowpointerB.in";
  opDesc.colpointer_matrix_b_init = "/worksnpace/sstStonne/tests/outerproduct/outerproduct_gemm_colpointerB.in";
  stonneCore->setup(opDesc);
  stonneCore->init(1);
};

SparseCore::~SparseCore() { delete stonneCore; }

bool SparseCore::running() {
  return !_request_queue.empty() || !_response_queue.empty() || !stonneCore->isFinished();
}

bool SparseCore::can_issue(const std::shared_ptr<Tile>& op) {
  return !running();
}

std::shared_ptr<Tile> SparseCore::pop_finished_tile() {
  return nullptr;
}

void SparseCore::cycle() {
  stonneCore->cycle();

  /* Send Memory Request */
  if (SimpleMem::Request* req = stonneCore->popRequest()) {
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

  if (!_response_queue.empty()) {
    mem_fetch* resp_wrapper = _response_queue.front();
    _request_queue.pop();
    SimpleMem::Request* resp = static_cast<SimpleMem::Request*>(resp_wrapper->get_custom_data());
    resp->setReply();
    stonneCore->pushResponse(resp);
    delete resp_wrapper;
  }
}

bool SparseCore::has_memory_request() {
  return !_request_queue.empty();
}

void SparseCore::pop_memory_request() {
  if (!_request_queue.empty()) {
    _request_queue.pop();
  }
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