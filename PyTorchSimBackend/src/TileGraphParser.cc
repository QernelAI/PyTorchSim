#include "TileGraphParser.h"

void printIndexMap(std::string prefix, const std::map<std::string, int>& indexMap) {
    std::ostringstream oss;
    for (const auto& [key, value] : indexMap) {
        oss << "{" << key << ": " << value << "} ";
    }
    spdlog::trace("{}: {}", prefix, oss.str());

}

uint32_t calculateAddress(const std::vector<uint32_t>& loop_size, const std::vector<uint32_t>& loop_idx) {
  std::vector<int> stride_list(loop_size.size(), 1);

  for (int i = loop_size.size() - 2; i >= 0; --i) {
    stride_list[i] = stride_list[i + 1] * loop_size[i + 1];
  }
  int address = std::inner_product(
      stride_list.begin(), stride_list.end(),
      loop_idx.begin(),
      0
  );
  return address;
}

std::vector<uint32_t> calc_output_idx(TileGraphParser* tog_parser, std::map<std::string, int>& iter) {
  // Extract outer loop
  // Extract inner loop
  std::vector<uint32_t> outer_loop;
  std::vector<uint32_t> inner_loop;
  uint32_t step = std::stoi(tog_parser->getMetaByName("systolic_size"));
  int offset = 0;
  for (auto loop_idx = iter.begin(); loop_idx != iter.end(); ++loop_idx) {
    if (tog_parser->get_loop_type(loop_idx->first)!=LoopType::INNER_LOOP)
      outer_loop.push_back(loop_idx->second);
    else
      inner_loop.push_back(loop_idx->second);
  }

  offset = outer_loop.size() - inner_loop.size();
  for (int i=0; i<inner_loop.size(); i++)
    outer_loop[offset+i] += inner_loop[i] * step;
  return outer_loop;
}

bool find_output_idx(TileGraphParser* tog_parser, std::vector<uint32_t>& output_idx) {
  if (output_idx.size() != 3) {
    spdlog::error("Unsupported type operation... Can't zero skip");
    exit(EXIT_FAILURE);
  }
  uint32_t m, n, k;
  m = output_idx.at(0);
  n = output_idx.at(1);
  k = output_idx.at(2);

  auto attr_json = tog_parser->get_attribute_file();

  // Check arg0: m -> k
  bool found_arg0 = false;
  if (attr_json["zero_skip"].contains("arg0")) {
    auto& arg0 = attr_json["zero_skip"]["arg0"];
    if (arg0.contains(std::to_string(m)) && arg0[std::to_string(m)].contains(std::to_string(k))) {
      found_arg0 = true;
    }
  }

  // Check arg1: n -> k
  bool found_arg1 = false;
  if (attr_json["zero_skip"].contains("arg1")) {
    auto& arg1 = attr_json["zero_skip"]["arg1"];
    if (arg1.contains(std::to_string(k)) && arg1[std::to_string(k)].contains(std::to_string(n))) {
      found_arg1 = true;
    }
  }
  return found_arg0 || found_arg1;
}

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
  else if (type == "memory_wait_node")
    return TileType::MEMORY_WAIT_NODE;
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
    if (attribute.name() == "torchsim_overlapping_cycle") {
      _overlapping_cycle = attribute.i();
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
    } else if (attribute.name() == "torchsim_tag_idx_list") {
      for (int i = 0; i < attribute.strings_size(); i++)
        _tag_idx_list.push_back(attribute.strings(i));
    } else if (attribute.name() == "torchsim_loop_idx_list") {
      for (int i = 0; i < attribute.strings_size(); i++)
        _loop_idx_list.push_back(attribute.strings(i));
    } else if (attribute.name() == "torchsim_is_async") {
      _is_async = attribute.i();
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
  spdlog::debug("{} tag_list: {}", spaces, fmt::join(_tag_idx_list, ", "));
  spdlog::debug("{} index_list: {}", spaces, fmt::join(_loop_idx_list, ", "));
}

TileMemoryWaitNode::TileMemoryWaitNode(onnx::NodeProto& node) : TileNode(node) {
  for (auto attribute : node.attribute()) {
    if (attribute.name() == "torchsim_tag_idx_list") {
      for (int i = 0; i < attribute.strings_size(); i++)
        _tag_idx_list.push_back(attribute.strings(i));
    } else if (attribute.name() == "torchsim_base_addr") {
      _base_addr_name = attribute.s();
    }
  }
}

void TileMemoryWaitNode::print_node() {
  TileNode::print_node();
  std::string spaces(get_depth(), '\t');
  spdlog::debug("{} tag_list: {}", spaces, fmt::join(_tag_idx_list, ", "));
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
      if (attribute.s() == "outer_loop") {
        _loop_type = LoopType::PARALLEL_LOOP;
      } else if (attribute.s() == "accumulation_loop") {
        _loop_type = LoopType::ACCUMULATION_LOOP;
      } else if (attribute.s() == "inner_loop") {
        _loop_type = LoopType::INNER_LOOP;
      } else {
        _loop_type = LoopType::NORMAL_LOOP;
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
      std::vector<std::string>& tag_idx_list = mem_node->get_tag_idx_list();
      std::vector<int> skip_idx_list;
      std::vector<int> values;
      bool skip = false;
      /* Find axis */
      if (mem_node->is_async_node()) {
        for (int i=0;i<tag_idx_list.size();i++) {
          if (tag_idx_list.at(i) == "0")
            skip_idx_list.push_back(i);
        }

        /* Extract iter values */
        std::transform(iter.begin(), iter.end(), std::back_inserter(values),
                    [](const std::pair<std::string, int>& pair) { return pair.second; });

        for (auto axis : skip_idx_list) {
          if (values.at(iter.size() - tag_idx_list.size() + axis) != 0) {
            skip = true;
            break;
          }
        }

        /* Skip this node */
        if (skip)
          continue;
      }

      /* Lookup given name's address */
      addr_type base_addr = tog_parser->lookup(base_addr_name);
      std::vector<int> iter_list;
      std::vector<int> tag_list;
      std::vector<int> loop_size_list;
      std::vector<uint32_t> outer_loop_idx;
      std::vector<uint32_t> outer_loop_size;
      int nr_inner_loop = 0;
      auto& loop_idx_list = mem_node->get_loop_idx_list();
      for (auto loop_idx: loop_idx_list) {
        auto iter_value = iter.at(loop_idx);
        iter_list.push_back(iter_value);
        loop_size_list.push_back(tog_parser->get_loop_size(loop_idx));
        if (tog_parser->get_loop_type(loop_idx)==LoopType::INNER_LOOP)
          nr_inner_loop++;
      }
      /* Add accumulation loop info to tag list */
      for (auto loop_idx = loop_idx_list.begin();
            loop_idx != loop_idx_list.end() - nr_inner_loop; ++loop_idx) {
        // Check loop type and process
        if (tog_parser->get_loop_type(*loop_idx)==LoopType::ACCUMULATION_LOOP) {
          auto iter_value = iter.at(*loop_idx);
          tag_list.push_back(iter_value);
        }
      }

      for (auto loop_idx = loop_idx_list.begin();
            loop_idx != loop_idx_list.end(); ++loop_idx) {
        if (tog_parser->get_loop_type(*loop_idx)==LoopType::PARALLEL_LOOP) {
          uint32_t step = (uint32_t)tog_parser->get_loop_step(*loop_idx);
          auto iter_value = iter.at(*loop_idx) / step;
          outer_loop_idx.push_back(iter_value);
          outer_loop_size.push_back(tog_parser->get_loop_size(*loop_idx));
        }
      }

      for (auto loop_idx: mem_node->get_tag_idx_list()) {
        if (iter.find(loop_idx) == iter.end())
          tag_list.push_back(0);
        else {
          auto iter_value = iter.at(loop_idx);
          tag_list.push_back(iter_value);
        }
      }

      /* Calc numa id */
      int numa_id = 0;
      auto numa_stride_size = tog_parser->lookupNumaInfo(base_addr_name).size();
      if (numa_stride_size) {
        int total_idx = calculateAddress(outer_loop_size, outer_loop_idx);
        int stride_idx = calculateAddress(outer_loop_size, tog_parser->lookupNumaInfo(base_addr_name));
        numa_id = total_idx / stride_idx;
      }

      printIndexMap("[TOGParser] Load Node " + mem_node->get_base_addr_name() + " Numa_id: " + std::to_string(numa_id), iter);
      std::shared_ptr<Instruction> inst = std::make_shared<Instruction>(
        Opcode::MOVIN, 0,
        0, base_addr,
        mem_node->get_tile_size(), mem_node->get_precision(), iter_list,
        mem_node->get_stride_list(), tag_list, loop_size_list
      );
      inst->set_addr_name(base_addr_name);
      inst->set_nr_inner_loop(nr_inner_loop);
      inst->adjust_dram_address();
      inst->set_is_async(mem_node->is_async_node());
      inst->set_numa_id(numa_id);
      link_map[tile_node] = inst;
      tile_vec.back()->append_instuction(inst);
    } else if (tile_node->get_type() == TileType::STORE_NODE) {
      std::shared_ptr<TileMemoryNode> mem_node = std::static_pointer_cast<TileMemoryNode>(tile_node);
      auto base_addr_name = mem_node->get_base_addr_name();
      /* Lookup given name's address */
      addr_type base_addr = tog_parser->lookup(base_addr_name);
      std::vector<int> iter_list;
      std::vector<int> loop_size_list;
      std::vector<uint32_t> outer_loop_idx;
      std::vector<uint32_t> outer_loop_size;
      int nr_inner_loop = 0;
      auto& loop_idx_list = mem_node->get_loop_idx_list();
      for (auto loop_idx: loop_idx_list) {
        auto iter_value = iter.at(loop_idx);
        iter_list.push_back(iter_value);
        loop_size_list.push_back(tog_parser->get_loop_size(loop_idx));
        if (tog_parser->get_loop_type(loop_idx)==LoopType::INNER_LOOP)
          nr_inner_loop++;
        if (tog_parser->get_loop_type(loop_idx)==LoopType::PARALLEL_LOOP) {
          uint32_t step = (uint32_t) tog_parser->get_loop_step(loop_idx);
          auto iter_value = iter.at(loop_idx) / step;
          outer_loop_idx.push_back(iter_value);
          outer_loop_size.push_back(tog_parser->get_loop_size(loop_idx)/ step);
        }
      }

      /* Calc numa id */
      int numa_id = 0;
      auto numa_stride_size = tog_parser->lookupNumaInfo(base_addr_name).size();
      if (numa_stride_size) {
        int total_idx = calculateAddress(outer_loop_size, outer_loop_idx);
        int stride_idx = calculateAddress(outer_loop_size, tog_parser->lookupNumaInfo(base_addr_name));
        numa_id = total_idx / stride_idx;
      }

      printIndexMap("[TOGParser] Store Node " + mem_node->get_base_addr_name() + " Numa_id: " + std::to_string(numa_id), iter);
      std::shared_ptr<Instruction> inst = std::make_shared<Instruction>(
        Opcode::MOVOUT, 0,
        0, base_addr,
        mem_node->get_tile_size(), mem_node->get_precision(), iter_list,
        mem_node->get_stride_list(), std::vector<int>(), loop_size_list
      );
      inst->set_addr_name(base_addr_name);
      inst->set_nr_inner_loop(nr_inner_loop);
      inst->adjust_dram_address();
      inst->set_is_async(mem_node->is_async_node());
      inst->set_numa_id(numa_id);
      link_map[tile_node] = inst;
      tile_vec.back()->append_instuction(inst);
    } else if (tile_node->get_type() == TileType::MEMORY_WAIT_NODE) {
      printIndexMap("[TOGParser] DMA Wait Node ", iter);
      std::shared_ptr<TileMemoryWaitNode> wait_node = std::static_pointer_cast<TileMemoryWaitNode>(tile_node);
      auto base_addr_name = wait_node->get_base_addr_name();
      addr_type base_addr = tog_parser->lookup(base_addr_name);
      /* Lookup given name's address */
      std::vector<int> iter_list;
      std::vector<int> tag_list;
      auto& wait_tag_list = wait_node->get_tag_idx_list();
      int inner_step = std::stoi(tog_parser->getMetaByName("systolic_size"));
      /* Add accumulation loop info to tag list */
      for (auto loop_idx = iter.begin(); loop_idx != iter.end(); ++loop_idx) {
        /* FIXME. Used heuristic that wait_tag_size has 2d dim */
        if (tog_parser->get_loop_type(loop_idx->first)==LoopType::ACCUMULATION_LOOP && wait_tag_list.size() != 2) {
          tag_list.push_back(loop_idx->second);
        }
      }

      for (auto loop_idx: wait_tag_list) {
        if (iter.find(loop_idx) == iter.end())
          tag_list.push_back(0);
        else {
          auto iter_value = iter.at(loop_idx) * inner_step;
          tag_list.push_back(iter_value);
        }
      }

      std::shared_ptr<Instruction> inst = std::make_shared<Instruction>(
        Opcode::BAR, 0,
        0, base_addr,
        std::vector<size_t>(), 0, iter_list,
        iter_list, tag_list, std::vector<int>()
      );
      inst->set_addr_name(base_addr_name);
      link_map[tile_node] = inst;
      tile_vec.back()->append_instuction(inst);
    } else if (tile_node->get_type() == TileType::COMPUTE_NODE) {
      printIndexMap("[TOGParser] Compute Node ", iter);
      std::shared_ptr<TileComputeNode> compute_node = std::static_pointer_cast<TileComputeNode>(tile_node);
      std::vector<int> iter_list;
      std::shared_ptr<Instruction> inst = std::make_shared<Instruction>(
        Opcode::COMP, compute_node->get_cycle(),
        0, 0,
        std::vector<size_t>(), 0, iter_list, iter_list,
        std::vector<int>(), std::vector<int>()
      );
      inst->set_overlapping_cycle(compute_node->get_overlapping_cycle());
      inst->set_compute_type(compute_node->get_compute_type());

      /* Check should we have to skip */
      auto output_idx_list = calc_output_idx(tog_parser, iter); // (M,N,K) order
      if (compute_node->get_compute_type() == 1 && output_idx_list.size() == 3) { // FIXME. hardcoded type
        bool skip = find_output_idx(tog_parser, output_idx_list);
        if (skip) {
          inst->set_compute_cycle(0);
          inst->set_overlapping_cycle(0);
          spdlog::trace("[TOGParser/Sparse] Skip output tile index: {}", fmt::join(output_idx_list, ","));
        }
      }

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

      std::map<std::string, int> inner_indices = iter;
      auto loop_type = loop_node->get_loop_type();
      auto& parent_instructions = parent->get_instructions();
      auto& last_instruction = parent_instructions.back();
      auto nr_inst = parent_instructions.size();
      for (int i=start; i<end; i+=stride) {
        inner_indices[loop_node->get_idx_name()] = i;
        std::vector<std::shared_ptr<Tile>> ret = loop_node->get_tiles_from_iter(tog_parser, inner_indices);
        if (loop_type == LoopType::INNER_LOOP) {
         for (const auto& inner_tile : ret) {
            for (auto& inner_inst : inner_tile->get_instructions()) {
              tile_vec.back()->append_instuction(inner_inst);
              if (nr_inst) {
                last_instruction->add_child(inner_inst);
              }
            }
          }
        } else {
          parent->append_child(ret.front());
          ret.back()->append_child(child);
          for (const auto& inner_tile : ret) {
            tile_vec.push_back(inner_tile);
          }
        }
      }
      /* Set last instruction's free sram size */
      if(parent->get_instructions().size())
        parent->get_instructions().back()->set_free_sram_size(parent->get_required_sram_size());

      parent->append_child(child);
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
      uint64_t value = it.value();
      _arg_to_address[it.key()] = value;
      spdlog::info("[TOGParser] Address Attribute key: {} address: 0x{:x}", it.key(), value);
    }
  }
  if (_attribute_json.contains("address_numa_stride")) {
    auto address_numa_stride = _attribute_json["address_numa_stride"];
    for (auto it = address_numa_stride.begin(); it != address_numa_stride.end(); ++it) {
      auto value_list = it.value();
      for (auto value : value_list) {
        _arg_numa_stride[it.key()].push_back(value);
      }
      spdlog::info("[TOGParser] Address numa info key: {} numa stride : {}", it.key(), fmt::join(_arg_numa_stride[it.key()], ", "));
    }
  }

  /* ONNX file parsing */
  _tog_path = onnx_path;
  model_proto.ParseFromZeroCopyStream(&zero_copy_input) && model_istream.eof();

  auto input = model_proto.graph().input();
  auto graph_name = model_proto.graph().name();
  graph_name = graph_name == "" ? "?" : graph_name;

  /* Get meta data from graph */
  for (const auto& meta : model_proto.metadata_props()) {
    spdlog::info("[TOGParser] Register Metadata \"{}\": \"{}\"", meta.key(), meta.value());
    _tog_meta[meta.key()] = meta.value();
  }

  for (onnx::NodeProto node_proto : model_proto.graph().node()) {
    std::string op_type = node_proto.op_type();
    TileType type = TileNode::get_tile_type(op_type);
    /* Parse node */
    if (type == TileType::LOOP_INDEX_NODE) {
      std::shared_ptr<TileLoopNode> tile_node = std::make_shared<TileLoopNode>(node_proto);

      /* Register output */
      register_tile(tile_node);
      register_loop(tile_node);
      increase_loop_top();

      /* Register loop info to parser */
      std::string loop_idx = tile_node->get_idx_name();
      uint64_t start = tile_node->get_start();
      uint64_t end = tile_node->get_end();
      uint64_t step = tile_node->get_stride();
      _loop_size_map[loop_idx] = std::tuple<int, int, LoopType>(end - start, step, tile_node->get_loop_type());
    } else if (type == TileType::LOOP_END_NODE) {
      std::shared_ptr<TileLoopEndNode> tile_node = std::make_shared<TileLoopEndNode>(node_proto);
      register_tile(tile_node);
      decrease_loop_top();
    } else if (type == TileType::LOAD_NODE || type == TileType::STORE_NODE) {
      std::shared_ptr<TileMemoryNode> tile_node = std::make_shared<TileMemoryNode>(node_proto);
      /* Register output */
      register_tile(tile_node);
    } else if (type == TileType::COMPUTE_NODE) {
      std::shared_ptr<TileComputeNode> tile_node = std::make_shared<TileComputeNode>(node_proto);
      /* Register output */
      register_tile(tile_node);
    } else if (type == TileType::MEMORY_WAIT_NODE) {
      std::shared_ptr<TileMemoryWaitNode> tile_node = std::make_shared<TileMemoryWaitNode>(node_proto);
      /* Register output */
      register_tile(tile_node);
    }
  }

  for (auto tile: _tile_vec) {
    if (tile->get_type() != TileType::LOOP_END_NODE)
      tile->print_node();
  }

  /* Generate subgraph */
  if (_loop_nodes.empty()) {
    spdlog::error("[TileGraphParser] No loop found...");
    exit(EXIT_FAILURE);
  }

  _tile_graph = std::make_unique<TileGraph>(TileGraph(onnx_path, graph_name));
  int last_outer_idx = -1;
  /* Extract outer loop */
  for (int i=0;i<_loop_nodes.size();i++) {
    std::shared_ptr<TileLoopNode> outer_loop = std::static_pointer_cast<TileLoopNode>(_loop_nodes.at(i).front());
    if (outer_loop->get_loop_type() != LoopType::PARALLEL_LOOP)
      break;
    last_outer_idx = i;
    std::string loop_idx = outer_loop->get_idx_name();
    uint64_t start = outer_loop->get_start();
    uint64_t end = outer_loop->get_end();
    uint64_t stride = outer_loop->get_stride();
    _tile_graph->push_range(loop_idx, {start, end, stride});
    spdlog::trace("[TOGParser] <Push Loop> loop_idx: {}, start: {}, end: {}, stride: {}", loop_idx, start, end, stride);
  }

  /* Iterate outer loop and initialize inner loop */
  for (auto iter=_tile_graph->begin(); iter!=_tile_graph->end(); ++iter) {
    std::shared_ptr<TileSubGraph> subgraph = std::make_shared<TileSubGraph>();
    subgraph->set_core_id(getCoreIdFromJson(attribute_json, subgraph->get_id()));
    auto indices = iter.get_indices();
    for (auto loop : _loop_nodes.at(last_outer_idx)) {
      std::shared_ptr<TileLoopNode> outer_loop = std::static_pointer_cast<TileLoopNode>(loop);
      std::vector<std::shared_ptr<Tile>> sub_tiles = outer_loop->get_tiles_from_iter(this, indices);

      /* insert tiles to subgraph */
      for (const auto& sub_tile: sub_tiles){
        subgraph->add_tile(sub_tile);
      }
    }
    /* insert subgraph to graph */
    _tile_graph->append_subgraph(subgraph);
  }
}

void TileGraphParser::register_loop(std::shared_ptr<TileNode> loop_node) {
  if (_loop_nodes.size() <= _loop_stack_pointer) {
    _loop_nodes.resize(_loop_stack_pointer + 1);
  }
  _loop_nodes.at(_loop_stack_pointer).push_back(loop_node);
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
  return _loop_nodes.at(_loop_stack_pointer-1).back();
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

const std::vector<uint32_t>& TileGraphParser::lookupNumaInfo(std::string key) {
  static std::vector<uint32_t> dummy_result = {};
  auto val = _arg_numa_stride.find(key);
  if (val == _arg_numa_stride.end())
    return dummy_result;
  return _arg_numa_stride.at(key);
}

int TileGraphParser::getCoreIdFromJson(const json& attribute_json, int subgraph_id) {
  if (attribute_json.contains("subgraph_map")) {
    const auto& subgraph_map = attribute_json["subgraph_map"];
    if (subgraph_map.contains(std::to_string(subgraph_id)) && subgraph_map[std::to_string(subgraph_id)].is_number_integer()) {
        return subgraph_map[std::to_string(subgraph_id)];
    }
  }
  return -1;
}