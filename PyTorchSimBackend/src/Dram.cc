#include "Dram.h"

uint32_t Dram::get_channel_id(mem_fetch* access) {
  uint32_t channel_id;
  if (_n_ch_per_partition >= 16)
    channel_id = ipoly_hash_function((new_addr_type)access->get_addr()/_config.dram_req_size, 0, _n_ch_per_partition);
  else
    channel_id = ipoly_hash_function((new_addr_type)access->get_addr()/_config.dram_req_size, 0, 16) % _n_ch_per_partition;

  channel_id += ((access->get_numa_id() % _n_partitions)* _n_ch_per_partition);
  return channel_id;
}

DramRamulator2::DramRamulator2(SimulationConfig config, cycle_type* core_cycle) {
  _core_cycles = core_cycle;
  _n_ch = config.dram_channels;
  _req_size = config.dram_req_size;
  _n_partitions = config.dram_num_partitions;
  _n_ch_per_partition = _n_ch / _n_partitions;
  _config = config;
  _mem.resize(_n_ch);

  spdlog::info("[Config/DRAM] DRAM Bandwidth {} GB/s, Freq: {} MHz, Channels: {}, Request_size: {}", config.max_dram_bandwidth(), config.dram_freq, _n_ch, _req_size);
  /* Initialize DRAM Channels */
  for (int ch = 0; ch < _n_ch; ch++) {
    m_to_crossbar_queue.push_back(std::queue<mem_fetch*>());
    m_from_crossbar_queue.push_back(std::queue<mem_fetch*>());
    _mem[ch] = std::make_unique<Ramulator2>(
      ch, _n_ch, config.dram_config_path, "Ramulator2", _config.dram_print_interval, 1);
  }

  /* Initialize L2 cache */
  _m_caches.resize(_n_ch);
  _m_cache_config.init(config.l2d_config_str);
  spdlog::info("[Config/L2] Total Size: {} KB, Partition Size: {} KB, Set: {}, Assoc: {}, Line Size: {}B Sector Size: {}B",
              _m_cache_config.get_total_size_in_kb() * _n_ch, _m_cache_config.get_total_size_in_kb(),
              _m_cache_config.get_num_sets(), _m_cache_config.get_num_assoc(),
              _m_cache_config.get_line_size(), _m_cache_config.get_sector_size());
  for (int ch = 0; ch < _n_ch; ch++) {
    m_to_mem_queue.push_back(std::queue<mem_fetch*>());
    m_cache_latency_queue.push_back(DelayQueue<mem_fetch*>("cache_latency_queue", true, 0));
    _m_caches[ch] = std::make_unique<ReadOnlyCache>("L2 RO cache", _m_cache_config, ch, 0, &m_to_mem_queue[ch]);
  }

  _tx_log2 = log2(_req_size);
  _tx_ch_log2 = log2(_n_ch_per_partition) + _tx_log2;
}

bool DramRamulator2::running() {
  return false;
}

void DramRamulator2::cache_cycle() {
  uint32_t line_size = _m_cache_config.get_line_size();
  uint32_t sector_size = _m_cache_config.get_sector_size();
  for (int i = 0; i < _n_ch; i++) {
    m_cache_latency_queue[i].cycle();
    // NDP to Cache. Read Only cache
    if (!m_from_crossbar_queue[i].empty() && !m_from_crossbar_queue[i].front()->is_write() &&
        _m_caches[i]->data_port_free()) {
      mem_fetch* req = m_from_crossbar_queue[i].front();
      req->set_access_sector_mask(line_size, sector_size);
      std::deque<CacheEvent> events;
      CacheRequestStatus status = _m_caches[i]->access(
          req->get_addr(), *_core_cycles, req, events);
      bool write_sent = CacheEvent::was_write_sent(events);
      bool read_sent = CacheEvent::was_read_sent(events);
      if (status == HIT) {
        if (!write_sent) {
          req->set_reply();
          m_cache_latency_queue[i].push(req, _config.l2d_hit_latency);
        }
        m_from_crossbar_queue[i].pop();
      } else if (status != RESERVATION_FAIL) {
        if (req->is_write() &&
            (_m_cache_config.get_write_alloc_policy() == FETCH_ON_WRITE ||
              _m_cache_config.get_write_alloc_policy() == LAZY_FETCH_ON_READ)) {
          req->set_reply();
          m_cache_latency_queue[i].push(req, _config.l2d_hit_latency);
        }
        m_from_crossbar_queue[i].pop();
      } else {
        // Status Reservation fail
        assert(!write_sent);
        assert(!read_sent);
      }
    }

    /* Write request is go mem directly */
    if(!m_from_crossbar_queue[i].empty() && m_from_crossbar_queue[i].front()->is_write()) {
      mem_fetch* req = m_from_crossbar_queue[i].front();
      m_to_mem_queue[i].push(req);
      m_from_crossbar_queue[i].pop();
    }

    if (_m_caches[i]->access_ready() &&
        !m_cache_latency_queue[i].full()) {
      mem_fetch* req = _m_caches[i]->top_next_access();
      req->current_state = "L2 top next access";
      if (req->is_request()) req->set_reply();
      m_cache_latency_queue[i].push(req, _config.l2d_hit_latency);
      _m_caches[i]->pop_next_access();
    }

    if (m_cache_latency_queue[i].arrived()) {
      mem_fetch* req = m_cache_latency_queue[i].top();
      m_to_crossbar_queue[i].push(req);
      m_cache_latency_queue[i].pop();
    }
    _m_caches[i]->cycle();
  }
}

void DramRamulator2::cycle() {
  for (int ch = 0; ch < _n_ch; ch++) {
    _mem[ch]->cycle();
    // From Cache to Ramulator
    if (!m_to_mem_queue[ch].empty()) {
      mem_fetch* mf = m_to_mem_queue[ch].front();
      _mem[ch]->push(mf);
      m_to_mem_queue[ch].pop();
    }
    // From memory response
    if (_mem[ch]->return_queue_top()) {
      mem_fetch* req = _mem[ch]->return_queue_top();
      if (_m_caches[ch]->waiting_for_fill(req)) {
        if (_m_caches[ch]->fill_port_free()) {
          _m_caches[ch]->fill(req, *_core_cycles);
          _mem[ch]->return_queue_pop();
        }
      } else {
        if (req->get_access_type() == L2_CACHE_WB &&
            req->get_type() == WRITE_ACK) {
          _mem[ch]->return_queue_pop();
          delete req;
        } else if (req->get_access_type() == GLOBAL_ACC_W &&
          req->get_type() == WRITE_ACK) {
          m_to_crossbar_queue[ch].push(req);
          _mem[ch]->return_queue_pop();
        }
      }
    }
  }
}

bool DramRamulator2::is_full(uint32_t cid, mem_fetch* request) {
  return false; //m_from_crossbar_queue[cid].full(); Infinite length
}

void DramRamulator2::push(uint32_t cid, mem_fetch* request) {
  m_from_crossbar_queue[cid].push(request);
}

bool DramRamulator2::is_empty(uint32_t cid) {
  return m_to_crossbar_queue[cid].empty();
}

mem_fetch* DramRamulator2::top(uint32_t cid) {
  assert(!is_empty(cid));
  return m_to_crossbar_queue[cid].front();
}

void DramRamulator2::pop(uint32_t cid) {
  assert(!is_empty(cid));
  m_to_crossbar_queue[cid].pop();
}

void DramRamulator2::print_stat() {
  for (int ch = 0; ch < _n_ch; ch++) {
    _mem[ch]->print(stdout);
  }
}
