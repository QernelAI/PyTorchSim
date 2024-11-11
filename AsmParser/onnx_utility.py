import onnx

class node:
    def __init__(self, node_id=0):
        self.id = node_id
        self.torchsim_name = self.__class__.__name__ + str(self.id)

        self.__parents = set()
        self.__children = set()
        self.inst = []

    def add_child(self, child):
        self.__children.add(child)

    def get_child(self):
        return list(self.__children)

    def add_parent(self, parent):
        self.__parents.add(parent)

    def get_parent(self):
        return list(self.__parents)

    def set_parent(self, parent):
        self.__parents = set(parent)

    def to_onnx(self):
        attr_dict = {}

        inputs = [p.torchsim_name + "_output" for p in self.__parents]
        outputs = [self.torchsim_name + "_output"]

        # Iterate all member variables
        for var in [attr for attr in dir(self) if not callable(getattr(self, attr)) and attr.startswith("torchsim")]:
            attr_dict[var] = getattr(self, var)

        inst_list = self.inst
        if len(self.inst) > 20:
            inst_list = self.inst[:10] + ["..."] + self.inst[-10:]

        for idx, asm_line in enumerate(inst_list):
            attr_dict[f"inst{idx:02x}"] = asm_line

        onnx_node = onnx.helper.make_node(op_type=self.__class__.__name__,
                                          inputs=inputs,
                                          outputs=outputs,
                                          **attr_dict)
        return onnx_node

class loop_index_node(node):
     def __init__(self, loop_idx, loop_info, node_id=0):
        super().__init__(node_id)
        self.torchsim_loop_idx = loop_idx
        self.torchsim_start = loop_info[0]
        self.torchsim_end = loop_info[1]
        self.torchsim_stride = loop_info[2]
        self.torchsim_loop_type = loop_info[3]

class loop_end_node(node):
    def __init__(self, loop_idx, node_id=0):
        super().__init__(node_id)
        self.torchsim_loop_idx = loop_idx

class memory_node(node):
    def __init__(self, tile_info, inst_list=list(), node_id=0):
        super().__init__(node_id)
        self.inst = inst_list
        self.torchsim_base_addr = tile_info["base_addr"]
        self.torchsim_stride_list = tile_info["stride_list"]
        self.torchsim_tile_size = tile_info["tile_size"]
        self.torchsim_tile_stride = tile_info["tile_stride"]
        self.torchsim_element_size = tile_info["element_size"]
        self.torchsim_tag_idx_list = tile_info["tag_idx_list"]
        self.torchsim_loop_idx_list = tile_info["loop_idx_list"]

class load_node(memory_node):
    pass

class store_node(memory_node):
    pass

class memory_wait_node(node):
    def __init__(self, tile_info, inst_list=list(), node_id=0):
        super().__init__(node_id)
        self.torchsim_tag_idx_list = tile_info["tag_idx_list"]
        self.torchsim_base_addr = tile_info["base_addr"]

class compute_node(node):
    def __init__(self, inst_list=list(), cycle=0, overlapping_cycle=0, compute_type=0, node_id=0):
        super().__init__(node_id)
        self.inst = inst_list
        self.torchsim_cycle = cycle
        self.torchsim_overlapping_cycle = overlapping_cycle
        self.torchsim_compute_type = compute_type

def connect_nodes(parent, child):
    child.add_parent(parent)
    parent.add_child(child)

def dump_onnx_graph(name, node_list):
    graph_def = onnx.helper.make_graph(
        inputs=[],
        outputs=[],
        nodes=node_list,
        name="Dummy tile graph",
    )
    model_def = onnx.helper.make_model(graph_def, producer_name="PyTorchSim")
    model_def.opset_import[0].version = 13

    onnx.save(model_def, name)

if __name__ == "__main__":
    load_node1 = load_node(0)
    load_node2 = load_node(1)
    compute_node1 = compute_node(2)
    store_node1 = store_node(3)

    loop_index_node1 = loop_index_node(node_id=0, start=[0,0,0], end=[1000,1000,1000], stride=[1,1,1])

    connect_nodes(loop_index_node1, load_node1)
    connect_nodes(loop_index_node1, load_node2)
    connect_nodes(loop_index_node1, store_node1)

    connect_nodes(load_node1, compute_node1)
    connect_nodes(load_node2, compute_node1)
    connect_nodes(compute_node1, store_node1)

    graph_def = onnx.helper.make_graph(
        inputs=[],#load_tile_name1, load_tile_name2],
        outputs=[],#store_tile_name],
        nodes=[loop_index_node1.to_onnx(), load_node1.to_onnx(), load_node2.to_onnx(), compute_node1.to_onnx(), store_node1.to_onnx()],
        name="Dummy tile graph",
    )
    model_def = onnx.helper.make_model(graph_def, producer_name="PyTorchSim")
    model_def.opset_import[0].version = 13

    onnx.save(model_def, "tile_graph.onnx")
