#pragma once
#include <fstream>
#include <algorithm>
#include <nlohmann/json.hpp>
#include <fmt/ranges.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include "TileGraph.h"
#include "Instruction.h"
#include "onnx/defs/schema.h"
#include "onnx/onnx-operators_pb.h"
#include "onnx/onnx_pb.h"

using json = nlohmann::json;

enum class TileType{
  LOOP_INDEX_NODE,
  LOOP_END_NODE,
  LOAD_NODE,
  STORE_NODE,
  COMPUTE_NODE,
  MEMORY_WAIT_NODE
};

enum class LoopType {
  NORMAL_LOOP,
  PARALLEL_LOOP,
  ACCUMULATION_LOOP,
  INNER_LOOP
};

class TileNode {
 public:
  TileNode(onnx::NodeProto& node);
  static TileType get_tile_type(std::string type);
  void add_child(std::shared_ptr<TileNode> child) { _child.push_back(std::move(child)); }
  std::vector<std::shared_ptr<TileNode>>& get_child() { return _child; }
  void add_parent(std::shared_ptr<TileNode> parent) { _parent.push_back(std::move(parent)); }
  std::vector<std::shared_ptr<TileNode>>& get_parent() { return _parent; }
  std::vector<std::string>& get_child_name() { return _child_name; }
  std::vector<std::string>& get_parent_name() { return _parent_name; }
  TileType get_type() { return _type; }
  std::shared_ptr<TileNode> get_owner_loop() { return _owner_loop; }
  std::string get_name() { return _name; }
  void set_owner_loop(std::shared_ptr<TileNode> owner) { _owner_loop=std::move(owner); }
  virtual void print_node();
  void set_depth(int depth) { _depth=depth; }
  int get_depth() { return _depth; }

 private:
  std::vector<std::shared_ptr<TileNode>> _parent;
  std::vector<std::shared_ptr<TileNode>> _child;
  std::vector<std::string> _parent_name;
  std::vector<std::string> _child_name;
  std::shared_ptr<TileNode> _owner_loop;
  std::string _name;
  int _depth;
  TileType _type;
};

class TileGraphParser {
 public:
  TileGraphParser(std::string onnx_path, json& attribute_json);
  std::shared_ptr<TileNode> get_top_loop();
  std::unique_ptr<TileGraph>& get_tile_graph() { return _tile_graph; }
  addr_type lookup(std::string key);
  void register_loop(std::shared_ptr<TileNode>);
  void increase_loop_top() { _loop_stack_pointer++; }
  void decrease_loop_top() { _loop_stack_pointer--; }
  int get_loop_size(std::string key) { return std::get<0>(_loop_size_map[key]); }
  int get_loop_step(std::string key) { return std::get<1>(_loop_size_map[key]); }
  LoopType get_loop_type(std::string key) { return std::get<2>(_loop_size_map[key]); }
  const std::map<std::string, std::tuple<int, int, LoopType>> & get_loop_map() { return _loop_size_map; }
  const std::vector<uint32_t> &lookupNumaInfo(std::string key);
  int getCoreIdFromJson(const json& attribute_json, int subgraph_id);
  std::string getMetaByName(std::string key) { return _tog_meta[key]; }
  const json& get_attribute_file() { return _attribute_json; }
 private:
  void register_tile(std::shared_ptr<TileNode> tile_node);
  void _tile_generate() {}
  void _base_addr_update() {}
  void _tile_index_generate() {}
  int _loop_stack_pointer = 0;

  json _attribute_json;
  std::string _tog_path;
  std::map<std::string, std::shared_ptr<TileNode>> _output_map;
  std::vector<std::vector<std::shared_ptr<TileNode>>> _loop_nodes;
  std::vector<std::shared_ptr<TileNode>> _tile_vec;
  std::unique_ptr<TileGraph> _tile_graph;
  std::map<std::string, addr_type> _arg_to_address;
  std::map<std::string, std::vector<uint32_t>> _arg_numa_stride;
  std::map<std::string, std::tuple<int, int, LoopType>> _loop_size_map;
  std::map<std::string, std::string> _tog_meta;
};

class TileComputeNode : public TileNode {
 public:
  TileComputeNode(onnx::NodeProto& node);
  uint32_t get_cycle() { return _cycle; }
  uint32_t get_overlapping_cycle() { return _overlapping_cycle; }
  int get_compute_type() { return _compute_type; }
  void print_node();

 private:
  std::map<std::string, std::shared_ptr<TileNode>> tile_map;
  uint32_t _cycle;
  uint32_t _overlapping_cycle = 0;
  int _compute_type;
};

class TileMemoryNode : public TileNode {
 public:
  TileMemoryNode(onnx::NodeProto& node);
  std::string get_base_addr_name() { return _base_addr_name; }
  size_t get_precision() { return _element_size; }
  std::vector<size_t> get_tile_size() { return _tile_size; }
  std::vector<int>& get_stride_list () { return _stride_list; }
  std::vector<std::string>& get_tag_idx_list() { return _tag_idx_list; }
  std::vector<std::string>& get_loop_idx_list() { return _loop_idx_list; }
  bool is_async_node() { return _is_async; }
  void print_node() override;

 private:
  std::vector<size_t> _tile_size;
  std::vector<int> _stride_list;
  size_t _element_size;
  bool _is_async;
  std::string _base_addr_name;
  std::vector<std::string> _tag_idx_list;
  std::vector<std::string> _loop_idx_list;
};

class TileMemoryWaitNode : public TileNode {
 public:
  TileMemoryWaitNode(onnx::NodeProto& node);
  std::string get_base_addr_name() { return _base_addr_name; }
  std::vector<std::string>& get_tag_idx_list() { return _tag_idx_list; }
  void print_node() override;

 private:
  std::vector<std::string> _tag_idx_list;
  std::string _base_addr_name;
};



class TileLoopNode : public TileNode {
 public:
 TileLoopNode(onnx::NodeProto& node);
  void add_body(std::shared_ptr<TileNode> body) { _body_node.push_back(body); }
  std::vector<std::shared_ptr<Tile>> get_tiles_from_iter(TileGraphParser*, std::map<std::string, int>&);
  std::string get_idx_name() { return _tile_index_name; }
  uint64_t get_start() { return _start; }
  uint64_t get_stride() { return _stride; }
  uint64_t get_end() { return _end; }
  LoopType get_loop_type() { return _loop_type; }
  void print_node() override;
 private:
  std::string _tile_index_name;
  uint64_t _stride;
  uint64_t _start;
  uint64_t _end;
  LoopType _loop_type;
  std::vector<std::shared_ptr<TileNode>> _body_node;
};

class TileLoopEndNode : public TileNode {
 public:
  TileLoopEndNode(onnx::NodeProto& node) : TileNode(node) {}
};
