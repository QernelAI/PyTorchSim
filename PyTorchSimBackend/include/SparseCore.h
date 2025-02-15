#include "Core.h"
#include "sstStonne.h"
#include "SimpleMem.h"

class SparseCore : public Core {
public:
  SparseCore(uint32_t id, SimulationConfig config);
  ~SparseCore() = default;
  bool running();
  bool can_issue(const std::shared_ptr<Tile>& op);
  std::shared_ptr<Tile> pop_finished_tile();
  void cycle();
  bool has_memory_request();
  void pop_memory_request();
  mem_fetch* top_memory_request() { return _request_queue.front(); }
  void push_memory_response(mem_fetch* response);
  void print_stats();
  void print_current_stats();

private:
  SST_STONNE::sstStonne *stonneCore;
  /* Interconnect queue */
  std::queue<mem_fetch*> _request_queue;
  std::queue<mem_fetch*> _response_queue;
};