#include "TileGraphParser.h"

TileNode::TileNode(onnx::NodeProto& node) {
  _type = get_tile_type(node.op_type());
  for (auto attribute : node.attribute()) {
    if (attribute.name() == "torchsim_name") {
      _name = attribute.s();
      break;
    }
  }

  /* insert input name */
  for (auto input : node.input()) {
    _parent_name.push_back(input);
  }

  /* insert output name */
  for (auto output : node.output()) {
    _child_name.push_back(output);
  }
}

TileType TileNode::get_tile_type(std::string type) {
  if (type == "loop_index_node")
    return TileType::LOOP_INDEX_NODE;
  else if (type == "loop_end_node")
    return TileType::LOOP_END_NODE;
  else if (type == "load_node")
    return TileType::LOAD_NODE;
  else if (type == "store_node")
    return TileType::STORE_NODE;
  else if (type == "compute_node")
    return TileType::COMPUTE_NODE;
  spdlog::error("[TileGraphParser] Invalid node type...");
  exit(EXIT_FAILURE);
}

void TileNode::print_node() {
  std::string spaces(_depth, '\t');
  spdlog::debug("{}Node type: {}, name: {}", spaces, int(_type), _name);
  spdlog::debug("{} input_name: {}", spaces,  _parent_name);
  spdlog::debug("{} output_name: {}", spaces, _child_name);

  for (auto& parent_ptr: _parent) {
    spdlog::debug("{} parent: {}", spaces, parent_ptr->get_name());
  }
  for (auto& child_ptr: _child) {
    spdlog::debug("{} child: {}", spaces, child_ptr->get_name());
  }
  if (_owner_loop != nullptr)
    spdlog::debug("{} owner: {}", spaces, _owner_loop->get_name());
  else
    spdlog::debug("{} owner: NULL", spaces);
}

TileComputeNode::TileComputeNode(onnx::NodeProto& node) : TileNode(node) {
  for (auto attribute : node.attribute()) {
    if (attribute.name() == "torchsim_cycle") {
      _cycle = int(attribute.i());
    }
    if (attribute.name() == "torchsim_compute_type") {
      _compute_type = attribute.i();
    }
  }
}

void TileComputeNode::print_node() {
  TileNode::print_node();
  std::string spaces(get_depth(), '\t');
  spdlog::debug("{} compute_cycle: {}", spaces, _cycle);
}

TileMemoryNode::TileMemoryNode(onnx::NodeProto& node) : TileNode(node) {
  for (auto attribute : node.attribute()) {
    if (attribute.name() == "torchsim_base_addr") {
      _base_addr_name = attribute.s();
    } else if (attribute.name() == "torchsim_element_size") {
      _element_size = attribute.i();
    } else if (attribute.name() == "torchsim_stride_list") {
      for (int i = 0; i < attribute.ints_size(); i++)
        _stride_list.push_back(attribute.ints(i));
    } else if (attribute.name() == "torchsim_tile_size") {
      for (int i = 0; i < attribute.ints_size(); i++)
        _tile_size.push_back(attribute.ints(i));
    } else if (attribute.name() == "torchsim_tile_stride") {
      for (int i = 0; i < attribute.ints_size(); i++)
        _tile_stride.push_back(attribute.ints(i));
    } else if (attribute.name() == "torchsim_loop_idx_list") {
      for (int i = 0; i < attribute.strings_size(); i++)
        _loop_idx_list.push_back(attribute.strings(i));
    }
  }
}

void TileMemoryNode::print_node() {
  TileNode::print_node();
  std::string spaces(get_depth(), '\t');
  spdlog::debug("{} base_addr_name: {}", spaces, _base_addr_name);
  spdlog::debug("{} element_size: {}", spaces, _element_size);
  spdlog::debug("{} stride_list: {} ", spaces, _stride_list);
  spdlog::debug("{} tile_size: {} ", spaces, _tile_size);
  spdlog::debug("{} tile_stride: {} ", spaces, _tile_stride);
}

TileLoopNode::TileLoopNode(onnx::NodeProto& node) : TileNode(node) {
  for (auto attribute : node.attribute()) {
    if (attribute.name() == "torchsim_start") {
      _start = attribute.i();
    } else if (attribute.name() == "torchsim_end") {
      _end = attribute.i();
    } else if (attribute.name() == "torchsim_stride") {
      _stride = attribute.i();
    } else if (attribute.name() == "torchsim_loop_idx") {
      _tile_index_name = attribute.s();
    } else if (attribute.name() == "torchsim_loop_type") {
      if (attribute.s() == "outer_loop")
        _loop_type = PARALLEL_LOOP;
      else if (attribute.s() == "accumulation_loop") {
        _loop_type = ACCUMULATION_LOOP;
      } else {
        _loop_type = NORMAL_LOOP;
      }
    }
  }
}

std::vector<std::shared_ptr<Tile>> TileLoopNode::get_tiles_from_iter(TileGraphParser* tog_parser, std::map<std::string, int>& iter) {
  std::vector<std::shared_ptr<Tile>> tile_vec;
  tile_vec.push_back(std::make_shared<Tile>(Tile::Status::INITIALIZED));

  std::map<std::shared_ptr<TileNode>, std::shared_ptr<Instruction>> link_map;
  for (auto& tile_node: _body_node) {
    if (tile_node->get_type() == TileType::LOAD_NODE) {
      std::shared_ptr<TileMemoryNode> mem_node = std::static_pointer_cast<TileMemoryNode>(tile_node);
      auto base_addr_name = mem_node->get_base_addr_name();
      /* Lookup given name's address */
      addr_type base_addr = tog_parser->lookup(base_addr_name);
      std::vector<int> iter_list;
      for (auto loop_idx: mem_node->get_loop_idx_list()) {
        auto iter_value = iter.at(loop_idx);
        iter_list.push_back(iter_value);
      }

      std::shared_ptr<Instruction> inst = std::make_shared<Instruction>(
        Opcode::MOVIN, 0,
        0, base_addr,
        mem_node->get_tile_size(), mem_node->get_tile_stride(), mem_node->get_precision(),
        iter_list
      );
      link_map[tile_node] = inst;
      tile_vec.back()->append_instuction(inst);
    } else if (tile_node->get_type() == TileType::STORE_NODE) {
      std::shared_ptr<TileMemoryNode> mem_node = std::static_pointer_cast<TileMemoryNode>(tile_node);
      auto base_addr_name = mem_node->get_base_addr_name();
      /* Lookup given name's address */
      addr_type base_addr = tog_parser->lookup(base_addr_name);
      std::vector<int> iter_list;
      for (auto loop_idx: mem_node->get_loop_idx_list()) {
        auto iter_value = iter.at(loop_idx);
        iter_list.push_back(iter_value);
      }

      std::shared_ptr<Instruction> inst = std::make_shared<Instruction>(
        Opcode::MOVOUT, 0,
        0, base_addr,
        mem_node->get_tile_size(), mem_node->get_tile_stride(), mem_node->get_precision(),
        iter_list
      );
      link_map[tile_node] = inst;
      tile_vec.back()->append_instuction(inst);
    } else if (tile_node->get_type() == TileType::COMPUTE_NODE) {
      std::shared_ptr<TileComputeNode> compute_node = std::static_pointer_cast<TileComputeNode>(tile_node);
      std::vector<int> iter_list;
      std::shared_ptr<Instruction> inst = std::make_shared<Instruction>(
        Opcode::COMP, compute_node->get_cycle(),
        0, 0,
        std::vector<size_t>(), std::vector<size_t>(), 0,
        iter_list
      );
      inst->set_compute_type(compute_node->get_compute_type());
      link_map[tile_node] = inst;
      tile_vec.back()->append_instuction(inst);
    } else if (tile_node->get_type() == TileType::LOOP_INDEX_NODE) {
      std::shared_ptr<TileLoopNode> loop_node = std::static_pointer_cast<TileLoopNode>(tile_node);
      uint64_t start = loop_node->get_start();
      uint64_t stride = loop_node->get_stride();
      uint64_t end = loop_node->get_end();

      /* Create tile before enter nested loop */
      for (const auto& pair: link_map) {
        std::shared_ptr<TileNode> node = pair.first;
        std::shared_ptr<Instruction> inst = pair.second;

        /* Link instruction dependency */
        for (const auto& child_node: node->get_child()) {
          if (link_map.find(child_node) != link_map.end()) {
            std::shared_ptr<Instruction> child_inst = link_map[child_node];
            inst->add_child(child_inst);
          }
        }
        /* Add instruction to tile */
        if (inst->get_opcode() == Opcode::MOVIN)
          tile_vec.back()->inc_required_sram_size(inst->get_tile_numel() * inst->get_precision());
      }
      link_map.clear();
      /* iterate nested loop */
      std::shared_ptr<Tile> parent = tile_vec.back();
      std::shared_ptr<Tile> child = std::make_shared<Tile>(Tile::Status::INITIALIZED);

      /* Set last instruction's free sram size */
      parent->get_instructions().back()->set_free_sram_size(parent->get_required_sram_size());

      std::map<std::string, int> inner_indices = iter;
      for (int i=start; i<end; i+=stride) {
        inner_indices[loop_node->get_idx_name()] = i;
        std::vector<std::shared_ptr<Tile>> ret = loop_node->get_tiles_from_iter(tog_parser, inner_indices);
        parent->append_child(ret.front());
        ret.back()->append_child(child);
        for (const auto& inner_tile : ret)
          tile_vec.push_back(inner_tile);
      }

      /* Create new tile */
      tile_vec.push_back(child);
    }
  }

  for (const auto& pair: link_map) {
    std::shared_ptr<TileNode> node = pair.first;
    std::shared_ptr<Instruction> inst = pair.second;

    /* Link instruction dependency */
    for (const auto& child_node: node->get_child()) {
      if (link_map.find(child_node) != link_map.end()) {
        std::shared_ptr<Instruction> child_inst = link_map[child_node];
        inst->add_child(child_inst);
      }
    }
    /* Add instruction to tile */
    if (inst->get_opcode() == Opcode::MOVIN)
      tile_vec.back()->inc_required_sram_size(inst->get_tile_numel() * inst->get_precision());
  }

  /* Set last instruction's free sram size */
  std::shared_ptr<Tile> parent = tile_vec.back();
  if (parent->get_instructions().size())
    parent->get_instructions().back()->set_free_sram_size(parent->get_required_sram_size());

  return tile_vec;
}

void TileLoopNode::print_node() {
  TileNode::print_node();
  std::string spaces(get_depth(), '\t');
  spdlog::debug("{} loop_idx: {} ", spaces, _tile_index_name);
  spdlog::debug("{} start: {} ", spaces, _start);
  spdlog::debug("{} end: {} ", spaces, _end);
  spdlog::debug("{} stride: {} ", spaces, _stride);
}

TileGraphParser::TileGraphParser(std::string onnx_path, json& attribute_json) {
  /* Note: this parsing algorithm assume that all node are sorted in topological-order */
  std::ifstream model_istream(onnx_path);
  google::protobuf::io::IstreamInputStream zero_copy_input(&model_istream);
  onnx::ModelProto model_proto;

  /* Attribute parsing */
  _attribute_json = attribute_json;
  if (_attribute_json.contains("address_info")) {
    auto address_info = _attribute_json["address_info"];
    for (auto it = address_info.begin(); it != address_info.end(); ++it) {
      int value = it.value();
      _arg_to_address[it.key()] = value;
      spdlog::info("[TOGPaser] Address Attribute key: {} address: 0x{:x}", it.key(), value);
    }
  }

  /* ONNX file parsing */
  _tog_path = onnx_path;
  model_proto.ParseFromZeroCopyStream(&zero_copy_input) && model_istream.eof();

  auto input = model_proto.graph().input();
  for (onnx::NodeProto node_proto : model_proto.graph().node()) {
    std::string op_type = node_proto.op_type();
    TileType type = TileNode::get_tile_type(op_type);
    /* Parse node */
    if (type == TileType::LOOP_INDEX_NODE) {
      std::shared_ptr<TileLoopNode> tile_node = std::make_shared<TileLoopNode>(node_proto);

      /* Register output */
      register_tile(tile_node);
      _loop_nodes.push_back(tile_node);
      _loop_stack_pointer++;
    } else if (type == TileType::LOOP_END_NODE) {
      std::shared_ptr<TileLoopEndNode> tile_node = std::make_shared<TileLoopEndNode>(node_proto);
      register_tile(tile_node);
      _loop_stack_pointer--;
    } else if (type == TileType::LOAD_NODE || type == TileType::STORE_NODE) {
      std::shared_ptr<TileMemoryNode> tile_node = std::make_shared<TileMemoryNode>(node_proto);
      /* Register output */
      register_tile(tile_node);
    } else if (type == TileType::COMPUTE_NODE) {
      std::shared_ptr<TileComputeNode> tile_node = std::make_shared<TileComputeNode>(node_proto);
      /* Register output */
      register_tile(tile_node);
    }
    //initialize_tile(node_proto.op_type());
  }

  for (auto tile: _tile_vec) {
    if (tile->get_type() != TileType::LOOP_END_NODE)
      tile->print_node();
  }

  /* Generate subgraph */
  if (_loop_nodes.size()==0) {
    spdlog::error("[TileGraphParser] No loop found...");
    exit(EXIT_FAILURE);
  }

  _tile_graph = std::make_unique<TileGraph>(TileGraph(onnx_path));
  int last_outer_idx = -1;
  /* Extract outer loop */
  for (int i=0;i<_loop_nodes.size();i++) {
    std::shared_ptr<TileLoopNode> outer_loop = std::static_pointer_cast<TileLoopNode>(_loop_nodes.at(i));
    if (outer_loop->get_loop_type() != TileLoopNode::PARALLEL_LOOP)
      break;
    last_outer_idx = i;
    std::string loop_idx = outer_loop->get_idx_name();
    uint64_t start = outer_loop->get_start();
    uint64_t end = outer_loop->get_end();
    uint64_t stride = outer_loop->get_stride();
    _tile_graph->push_range(loop_idx, {start, end, stride});
  }

  /* Iterate outer loop and initialize inner loop */
  for (auto iter=_tile_graph->begin(); iter!=_tile_graph->end(); ++iter) {
    std::shared_ptr<TileLoopNode> outer_loop = std::static_pointer_cast<TileLoopNode>(_loop_nodes.at(last_outer_idx));
    auto indices = iter.get_indices();
    std::shared_ptr<TileSubGraph> subgraph = std::make_shared<TileSubGraph>();
    std::vector<std::shared_ptr<Tile>> sub_tiles = outer_loop->get_tiles_from_iter(this, indices);

    /* insert tiles to subgraph */
    for (const auto& sub_tile: sub_tiles){
      subgraph->add_tile(sub_tile);
    }
    /* insert subgraph to graph */
    _tile_graph->append_subgraph(subgraph);
  }
}

void TileGraphParser::register_tile(std::shared_ptr<TileNode> tile_node) {
  tile_node->set_depth(_loop_stack_pointer);
  /* register output */
  for (std::string output_name : tile_node->get_child_name()) {
    _output_map[output_name] = tile_node;
  }

  /* register tile vec*/
  _tile_vec.push_back(tile_node);

  /* Update owner loop tile */
  tile_node->set_owner_loop(get_top_loop());
  std::shared_ptr<TileLoopNode> owner = std::static_pointer_cast<TileLoopNode>(tile_node->get_owner_loop());
  if (owner != nullptr) {
    owner->add_body(tile_node);
  }

  /* Skip loop end node */
  if (tile_node->get_type() == TileType::LOOP_END_NODE)
    return;

  /* Link parent tile */
  for (std::string input_name : tile_node->get_parent_name()) {
    std::shared_ptr<TileNode> parent = _output_map[input_name];
    if (parent->get_type() == TileType::LOOP_END_NODE) {
      parent->get_owner_loop()->add_child(tile_node);
      tile_node->add_parent(parent->get_owner_loop());
    } else if (parent->get_type() != TileType::LOOP_INDEX_NODE) {
      parent->add_child(tile_node);
      tile_node->add_parent(parent);
    }
  }
}

std::shared_ptr<TileNode> TileGraphParser::get_top_loop() {
  if (_loop_nodes.empty())
    return nullptr;
  return _loop_nodes.at(_loop_stack_pointer-1);
}

addr_type TileGraphParser::lookup(std::string key) {
  try {
    return _arg_to_address.at(key);
  } catch (const std::out_of_range& e) {
    spdlog::warn("[TOGParser] Key not found {} in the \"{}\"", key, _tog_path);
    _arg_to_address[key] = 0;
    return 0;
  }
}