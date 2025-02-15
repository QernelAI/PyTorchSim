#ifndef _TILE_H
#define _TILE_H

#include <memory>
#include <deque>
#include "Instruction.h"

class TileSubGraph;

class Tile {
 public:
  enum class Status {
    INITIALIZED,
    RUNNING,
    FINISH,
    EMPTY,
  };

  Tile(Status status);
  std::shared_ptr<TileSubGraph> get_owner() { return _onwer_graph; }
  void set_owner(std::shared_ptr<TileSubGraph> graph) { _onwer_graph = graph; }
  Status get_status() { return _status; }
  void set_status(Status status) { _status=status; }
  size_t get_ready_counter() { return _ready_counter; }
  void inc_ready_counter(); 
  void dec_ready_counter(); 
  size_t get_required_sram_size() { return _required_sram_size; }
  void set_required_sram_size(size_t sram_size) { _required_sram_size=sram_size; }
  void inc_required_sram_size(size_t sram_size) { _required_sram_size+=sram_size; }
  void append_instuction(std::shared_ptr<Instruction>& inst);
  void append_child(std::shared_ptr<Tile> child);
  std::vector<std::shared_ptr<Tile>>& get_child_tile () { return _child_tiles; }
  void finish_tile();
  bool is_ready() { return _ready_counter==0; }
  std::deque<std::shared_ptr<Instruction>>& get_instructions() { return _instructions; } 
  void print();
  size_t nr_insts() { return _nr_insts; }
  size_t nr_finshed_insts() { return _nr_finished_insts; }
  void inc_finished_inst() {
    _nr_finished_insts++;
  };
  bool all_insts_finshed() { return _nr_insts == _nr_finished_insts; }
  
 protected:
  std::shared_ptr<TileSubGraph> _onwer_graph;
  Status _status = Status::EMPTY;
  size_t _required_sram_size=0;
  size_t _ready_counter=0;
  size_t _nr_insts = 0;
  size_t _nr_finished_insts = 0;
  std::deque<std::shared_ptr<Instruction>> _instructions;
  std::vector<std::shared_ptr<Tile>> _child_tiles;
};

#endif