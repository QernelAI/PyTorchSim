#include <map>
#include <vector>
#include "Core.h"
#include "sstStonne.h"
#include "SimpleMem.h"
#include "Config.h"

class SparseCore : public Core {
public:
  SparseCore(uint32_t id, SimulationConfig config);
  ~SparseCore();
  bool running() override;
  bool can_issue(const std::shared_ptr<Tile>& op) override;
  void issue(std::shared_ptr<Tile> tile) override;
  void cycle() override;
  bool has_memory_request();
  void pop_memory_request();
  mem_fetch* top_memory_request() { return _request_queue.front(); }
  void push_memory_response(mem_fetch* response) override;
  void print_stats() override;
  void print_current_stats() override;

  uint32_t r_port_nr = 1;
  uint32_t w_port_nr = 1;
private:
  SST_STONNE::sstStonne *stonneCore;
  /* Interconnect queue */
  std::queue<mem_fetch*> _request_queue;
  std::queue<mem_fetch*> _response_queue;
  std::map<std::tuple<uint64_t, mem_access_type, mf_type>, std::vector<SimpleMem::Request*>*> request_merge_table;
};