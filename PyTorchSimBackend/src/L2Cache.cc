#include "L2Cache.h"

bool NoL2Cache::push(mem_fetch* req) {
  l_to_xbar_queue->push(req);
  return true;
}
void NoL2Cache::cycle() {
  if (!l_from_xbar_queue->empty()) {
    mem_fetch* req = l_from_xbar_queue->front();
    l_to_mem_queue.push(req);
    l_from_xbar_queue->pop();
  }
}

ReadOnlyL2Cache::ReadOnlyL2Cache(std::string name,  CacheConfig &cache_config, uint32_t id,
  cycle_type *core_cycle, uint32_t l2d_hit_latency,
  std::queue<mem_fetch*> *to_xbar_queue, std::queue<mem_fetch*> *from_xbar_queue) :
  L2Cache(name, cache_config, id, core_cycle, l2d_hit_latency, to_xbar_queue, from_xbar_queue) {
  l_cache = std::make_unique<ReadOnlyCache>(name, cache_config, id, 0, &l_to_mem_queue);
  l_from_cache_queue = DelayQueue<mem_fetch*>(l_name + "_latency_queue", true, 0);
}

bool ReadOnlyL2Cache::push(mem_fetch* req) {
  if (l_cache->waiting_for_fill(req)) {
    if (!l_cache->fill_port_free())
      return false;
    l_cache->fill(req, *l_core_cycle);
  } else {
    if (req->get_access_type() == L2_CACHE_WB && req->get_type() == WRITE_ACK) {
      delete req;
    } else if (req->get_access_type() == GLOBAL_ACC_W && req->get_type() == WRITE_ACK) {
      l_to_xbar_queue->push(req);
    }
  }
  return true;
}

void ReadOnlyL2Cache::cycle() {
  l_from_cache_queue.cycle();
  l_cache->cycle();

  // Mem to Cache. Read Only cache
  uint32_t line_size = l_cache_config.get_line_size();
  uint32_t sector_size = l_cache_config.get_sector_size();

  /* Read request*/
  if (!l_from_xbar_queue->empty() && !l_from_xbar_queue->front()->is_write() &&
        l_cache->data_port_free()) {
    mem_fetch* req = l_from_xbar_queue->front();
    req->set_access_sector_mask(line_size, sector_size);
    std::deque<CacheEvent> events;
    CacheRequestStatus status = l_cache->access(
        req->get_addr(), *l_core_cycle, req, events);
    bool write_sent = CacheEvent::was_write_sent(events);
    bool read_sent = CacheEvent::was_read_sent(events);
    if (status == HIT) {
      if (!write_sent) {
        req->set_reply();
        l_from_cache_queue.push(req, l2d_hit_latency);
      }
      l_from_xbar_queue->pop();
    } else if (status != RESERVATION_FAIL) {
      if (req->is_write() && // FIXME: req->is_write() already checked above 48 line.
          (l_cache_config.get_write_alloc_policy() == FETCH_ON_WRITE ||
            l_cache_config.get_write_alloc_policy() == LAZY_FETCH_ON_READ)) {
        req->set_reply();
        l_from_cache_queue.push(req, l2d_hit_latency);
      }
      l_from_xbar_queue->pop();
    } else {
      // Status Reservation fail
      assert(!write_sent);
      assert(!read_sent);
    }
  }

  /* Write request is go mem directly */
  if(!l_from_xbar_queue->empty() && l_from_xbar_queue->front()->is_write()) {
    mem_fetch* req = l_from_xbar_queue->front();
    l_to_mem_queue.push(req);
    l_from_xbar_queue->pop();
  }

  if (l_cache->access_ready() &&
      !l_from_cache_queue.full()) {
    mem_fetch* req = l_cache->top_next_access();
    req->current_state = "L2 top next access";
    if (req->is_request()) req->set_reply();
    l_from_cache_queue.push(req, l2d_hit_latency);
    l_cache->pop_next_access();
  }

  if (l_from_cache_queue.arrived()) {
    mem_fetch* req = l_from_cache_queue.top();
    l_to_xbar_queue->push(req);
    l_from_cache_queue.pop();
  }
}

void ReadOnlyL2Cache::print_stats() {
  if (l_id == 0) {
    l_cache->get_stats().print_stats(stdout, l_name.c_str());
  }
}