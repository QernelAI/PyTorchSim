import os
import sys
import importlib.util
from pathlib import Path
from collections import defaultdict

if __name__ == "__main__":
    from onnx_utility import node, loop_index_node, loop_end_node, load_node, store_node, compute_node, connect_nodes, dump_onnx_graph
else:
    from AsmParser.onnx_utility import node, loop_index_node, loop_end_node, load_node, store_node, compute_node, connect_nodes, dump_onnx_graph


def import_module_from_path(module_name, path):
    module_path = Path(path)  # Convert to Path object for safety
    if not module_path.exists() or not module_path.is_file():
        raise FileNotFoundError(f"No such file: '{module_path}'")

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None:
        raise ImportError(f"Could not load module from path: '{module_path}'")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module

class tog_generator:
    BaseNodeKind = 0
    ComputeNodeKind = 1
    LoopNodeKind = 2
    DMANodeKind = 3
    def __init__(self) -> None:
        self.module_name = "tile_operation_graph"
        self.module = None
        self.raw_graph = {}
        self.node_depth_stack = [[]]
        self.node_dict = {}
        self.parent_to_children = defaultdict(list)
        self.new_node_id = 0
        self.loop_end_stack = []

    def append_depth_stack(self, node):
        self.node_depth_stack[-1].append(node)

    def increase_depth_stack(self):
        self.node_depth_stack.append([])

    def load_file(self, path):
        self.module = import_module_from_path(self.module_name, path)
        self.raw_graph = self.module.graph
        self.parse_graph()

    def create_node(self, dump_data, prev_node):
        node_id = dump_data["node_id"]
        node_type = dump_data["node_type"]
        if node_type == self.BaseNodeKind:
            new_node = node(node_id)
        elif node_type == self.ComputeNodeKind:
            cycle = dump_data["compute_cycle"]
            compute_type = dump_data["compute_type"]
            new_node = compute_node(cycle=cycle, compute_type=compute_type, node_id=node_id)
        elif node_type == self.LoopNodeKind:
            loop_start = dump_data["loop_start"]
            loop_end = dump_data["loop_end"]
            loop_step  = dump_data["loop_step"]
            loop_idx = dump_data["loop_index"]
            loop_type = dump_data["loop_type"]
            new_node = loop_index_node(loop_idx, [loop_start, loop_end, loop_step, loop_type], node_id)

            # Push end node to stack
            new_end_node = loop_end_node(loop_idx, self.new_node_id)
            new_end_node.parent = dump_data["parents"][0]

            self.loop_end_stack.append(new_end_node)
            self.new_node_id += 1
            # Increase loop depth
            self.increase_depth_stack()
        elif node_type == self.DMANodeKind:
            tile_info = {}
            tile_info["base_addr"] = dump_data["base_address"]
            tile_info["stride_list"] = dump_data["stride_list"]
            tile_info["tile_stride"] = dump_data["tile_stride"]
            tile_info["tile_size"] = dump_data["tile_size"]
            tile_info["element_size"] = dump_data["element_size"]
            is_write = dump_data["is_write"]
            if is_write:
                new_node = store_node(tile_info, node_id=node_id)
            else:
                new_node = load_node(tile_info, node_id=node_id)
        else:
            print("Unexpected node_type :", node_type)
            exit(1)
        # add new meta data
        if node_id == 0:
            new_node.parent = -1
        else:
            new_node.parent = dump_data["parents"][0]
        self.append_depth_stack(new_node)

        # Return
        if not prev_node:
            self.node_dict[new_node.id] = new_node
            return new_node

        if prev_node[-1].parent == new_node.parent:
            # Handle special cases
            if isinstance(prev_node[-1], load_node) and isinstance(new_node, load_node):
                connect_nodes(prev_node[-1].get_parent()[-1], new_node)
            elif isinstance(prev_node[-1], store_node) and isinstance(new_node, store_node):
                connect_nodes(prev_node[-1].get_parent()[-1], new_node)
            elif isinstance(prev_node[-1], load_node) and isinstance(new_node, compute_node):
                for pn in prev_node:
                    if isinstance(pn, load_node):
                        connect_nodes(pn, new_node)
            else:
                connect_nodes(prev_node[-1], new_node)
        elif prev_node[-1].id == new_node.parent:
            connect_nodes(prev_node[-1], new_node)
        else:
            end_node = self.loop_end_stack.pop()
            self.node_dict[end_node.id] = end_node
            connect_nodes(prev_node[-1], end_node)
            connect_nodes(end_node, new_node)
        self.node_dict[new_node.id] = new_node
        return new_node

    def parse_graph(self):
        # Create nodes
        prev_node = []
        self.new_node_id = len(self.raw_graph.values()) + 1
        for value in self.raw_graph.values():
            new_node = self.create_node(value, prev_node)
            if not prev_node or prev_node[-1].parent == new_node.parent:
                prev_node.append(new_node)
            else:
                prev_node = [new_node]

        prev_node = prev_node[-1]
        # Link remain end node
        while self.loop_end_stack:
            end_node = self.loop_end_stack.pop()
            self.node_dict[end_node.id] = end_node
            connect_nodes(prev_node, end_node)
            prev_node = end_node

    def generate_tile_graph(self, name="tile_graph", cycle_list=list):
        node_list = list(self.node_dict.values())[1:]
        node_list[0].set_parent([])
        for iter_node in self.node_dict.values():
            if isinstance(iter_node, compute_node):
                iter_node.torchsim_cycle = cycle_list.pop(0)

        onnx_node_list = [node.to_onnx() for node in node_list] # Exclude root node
        dump_onnx_graph(name, onnx_node_list)

if __name__ == "__main__":
    t = tog_generator()
    t.load_file("/workspace/llvm-project/build/tile_operation_graph.py")
    t.parse_graph()