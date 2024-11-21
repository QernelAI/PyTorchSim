#include "TileGraph.h"

int TileSubGraph::_next_id = 0;
TileSubGraph::TileSubGraph() : _ready_tile_queue(), _tile_set(), _id(_next_id++) {
}

void TileSubGraph::add_tile(std::shared_ptr<Tile> tile) {
  tile->set_ownwer(this);
  for (auto& inst : tile->get_instructions())
    inst->subgraph_id = _id;
  if (tile->get_ready_counter() == 0) {
   _ready_tile_queue.push(tile);
  } else {
    _tile_set.insert(tile);
  }
}

void TileSubGraph::finish_tile(std::shared_ptr<Tile> tile) {
  /* TODO. */
  tile->finish_tile();
  for (auto child_tile_ptr: tile->get_child_tile()) {
    if (child_tile_ptr->get_ready_counter())
      continue;
    /* if child is ready, add ready queue */
    _ready_tile_queue.push(child_tile_ptr);
    _tile_set.erase(child_tile_ptr);
  }
  return;
}

const std::shared_ptr<Tile> TileSubGraph::peek_tile() {
  std::shared_ptr<Tile> ret = std::make_shared<Tile>(Tile::Status::EMPTY);
  if (_ready_tile_queue.empty())
    return ret;
  return _ready_tile_queue.top();
}

std::shared_ptr<Tile> TileSubGraph::get_tile() {
  if (_ready_tile_queue.empty()) {
    std::shared_ptr<Tile> ret = std::make_shared<Tile>(Tile::Status::EMPTY);
    return ret;
  } else {
    std::shared_ptr<Tile> ret = _ready_tile_queue.top();
    _ready_tile_queue.pop();
    return ret;
  }
}


void TileGraph::append_subgraph(std::shared_ptr<TileSubGraph> subgraph) {
  if(subgraph->get_core_id() == -1){
    _subgraph_vec.push_back(std::move(subgraph));
  } 
  else {
    _mapped_subgraph[subgraph->get_core_id()].push(std::move(subgraph));
  }
}

bool TileGraph::is_finished() {
  bool finished = _vec_index == _subgraph_vec.size() && _mapped_subgraph.empty();
  /* Check all outer loop is allocated */
  if (!finished)
    return finished;

  /* Check allocated subgraph is finished */
  for (const auto& core_pair: _cpu_graph_map) {
    for (const auto& tile_pair: core_pair.second)
      if (tile_pair.second != nullptr)
        finished &= tile_pair.second->is_finished();
  }

  return finished;
}

const std::shared_ptr<Tile> TileGraph::peek_tile(int core_id, int slot_id) {
  std::shared_ptr<Tile> ret = std::make_unique<Tile>(Tile(Tile::Status::EMPTY));
  if (_cpu_graph_map.find(core_id) == _cpu_graph_map.end()) {
    allocate_subgraph(core_id, slot_id);
    return ret;
  } else if (_cpu_graph_map[core_id].find(slot_id) == _cpu_graph_map[core_id].end()) {
    allocate_subgraph(core_id, slot_id);
    return ret;
  } else if (_cpu_graph_map[core_id][slot_id] == nullptr) {
    allocate_subgraph(core_id, slot_id);
    return ret;
  }

  if (_cpu_graph_map[core_id][slot_id]->is_finished()){
    allocate_subgraph(core_id, slot_id);
    return ret;
  }
  return _cpu_graph_map[core_id][slot_id]->peek_tile();
}

std::shared_ptr<Tile> TileGraph::get_tile(int core_id, int slot_id) {
  std::shared_ptr<Tile> ret = std::make_unique<Tile>(Tile(Tile::Status::EMPTY));
  if (_cpu_graph_map.find(core_id) == _cpu_graph_map.end()) {
    allocate_subgraph(core_id, slot_id);
    return ret;
  } else if (_cpu_graph_map[core_id].find(slot_id) == _cpu_graph_map[core_id].end()) {
    allocate_subgraph(core_id, slot_id);
    return ret;
  }

  if (_cpu_graph_map[core_id][slot_id]->is_finished()) {
    allocate_subgraph(core_id, slot_id);
    return ret;
  }
  return _cpu_graph_map[core_id][slot_id]->get_tile();
}

void TileGraph::allocate_subgraph(int core_id, int slot_id) {
  bool mapped_sub_graph_empty = true;
  if (_mapped_subgraph.find(core_id) != _mapped_subgraph.end() && !_mapped_subgraph[core_id].empty()) {
    mapped_sub_graph_empty = false;
    _cpu_graph_map[core_id][slot_id] = _mapped_subgraph[core_id].front();
    _mapped_subgraph[core_id].pop();
    if(_mapped_subgraph[core_id].empty())
      _mapped_subgraph.erase(core_id);
  }
  else if (_vec_index !=_subgraph_vec.size()) {
    spdlog::trace("[TileGraph] Core {} allocated new subgraph ({}/{})", core_id, _vec_index, _subgraph_vec.size());
    std::shared_ptr<TileSubGraph> subgraph = _subgraph_vec.at(_vec_index++);
    _cpu_graph_map[core_id][slot_id] = subgraph;
  }
  else {
    _cpu_graph_map[core_id][slot_id] = nullptr;
  }
}