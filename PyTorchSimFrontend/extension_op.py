import os
import subprocess
import math
import struct
import torch
import numpy as np
from torch._inductor.select_algorithm import ExternKernelChoice
from AsmParser.tog_generator import tog_generator
from torch._inductor.codecache import write
from PyTorchSimFrontend.extension_codecache import get_write_path
from PyTorchSimFrontend import extension_config
from Simulator.simulator import BackendSimulator, TORCH_TO_NUMPY

class MLIRExternKernelChoice(ExternKernelChoice):
    def call_name(self):
        return f"torch.ops.extension_op.{self.name}"

custom_lib = torch.library.Library("extension_op", "DEF")

def generate_outer_product_matrix(a, b, M, K, N):
    # Generating matrix A
    data_width = 4
    a_cpu = a.cpu()
    b_cpu = b.cpu()
    value_pointer = os.path.join(extension_config.CONFIG_TORCHSIM_DIR,
        'PyTorchSimBackend/extern/stonneCore/tests/outerproduct/outerproduct_gemm_mem.ini')
    rowA_pointer = os.path.join(extension_config.CONFIG_TORCHSIM_DIR,
        f'PyTorchSimBackend/extern/stonneCore/tests/outerproduct/outerproduct_gemm_rowpointerA.in')
    colA_pointer = os.path.join(extension_config.CONFIG_TORCHSIM_DIR,
        f'PyTorchSimBackend/extern/stonneCore/tests/outerproduct/outerproduct_gemm_colpointerA.in')
    rowB_pointer = os.path.join(extension_config.CONFIG_TORCHSIM_DIR,
        f'PyTorchSimBackend/extern/stonneCore/tests/outerproduct/outerproduct_gemm_rowpointerB.in')
    colB_pointer = os.path.join(extension_config.CONFIG_TORCHSIM_DIR,
        f'PyTorchSimBackend/extern/stonneCore/tests/outerproduct/outerproduct_gemm_colpointerB.in')
    with open(value_pointer, "w") as fd, open(rowA_pointer, "w") as rpA, open(colA_pointer, "w") as cpA, open(rowB_pointer, "w") as rpB, open(colB_pointer, "w") as cpB:
        #generating matrixA
        n_nonzeros=0
        for k in range(K):  # col major
            initial_values=0
            rpA.write(str(n_nonzeros)+","); # writing the index of A
            for m in range(M):
                if(a_cpu[m, k]):  # value is nonzero
                    if((m==(M-1)) and (k==(K-1))):
                        cpA.write(str(m))
                    else:
                        cpA.write(str(m)+","); #writing the row index
                    initial_values+=1
                    value = a_cpu[m, k]
                    ba = bytearray(struct.pack(">f", value))  # generating list of bytes
                    my_int = int.from_bytes(ba, "big")
                    fd.write(str(my_int))
                    fd.write(",")
                    n_nonzeros+=1
        rpA.write(str(n_nonzeros))
        address_matrix_b=n_nonzeros*data_width
        #Generating matrix B
        n_nonzeros=0
        for k in range(0,K):  # Row major
            initial_values=0
            rpB.write(str(n_nonzeros)+","); # writing the index of A
            for n in range(0,N):
                if(b_cpu[k, n]):  # value is nonzero
                    if((k==(K-1)) and (n==(N-1))):
                        cpB.write(str(n))
                    else:
                        cpB.write(str(n)+","); #writing the row index

                    initial_values+=1
                    value = b_cpu[k, n]
                    ba = bytearray(struct.pack(">f", value))  # generating list of bytes
                    my_int = int.from_bytes(ba, "big")
                    fd.write(str(my_int))
                    fd.write(",")
                    n_nonzeros+=1

        rpB.write(str(n_nonzeros))
        fd.write(str(0)) # Adding a final 0 to the memory which will never be used. This is just to avoid having a last comma.
        address_matrix_c=address_matrix_b+(n_nonzeros*data_width)
    return 0, address_matrix_b, address_matrix_c

def flexagon_frontend(a, b, out):
    print("FLEXAGON FRONTEND")
    M = a.shape[0]
    N = b.shape[1]
    K = b.shape[0]

    def calculate_sparsity(tensor):
        total_elements = tensor.numel()
        zero_elements = torch.sum(tensor.cpu() == 0)
        sparsity_ratio = zero_elements / total_elements * 100
        return math.ceil(sparsity_ratio.item())

    x_sparsity = calculate_sparsity(a)
    w_sparsity = calculate_sparsity(b)
    assert(x_sparsity >= 0 and x_sparsity < 100)
    assert(w_sparsity >= 0 and w_sparsity < 100)
    print(f"A Sparsity: {x_sparsity}")
    print(f"B Sparsity: {w_sparsity}")

    # Generating inputs
    dir_path = os.path.join(
        extension_config.CONFIG_TORCHSIM_DIR,
        'PyTorchSimBackend/extern/stonneCore/tests/outerproduct'
    )
    os.makedirs(dir_path, exist_ok=True)

    value_path = os.path.join(
        extension_config.CONFIG_TORCHSIM_DIR,
        'PyTorchSimBackend/extern/stonneCore/tests/outerproduct/outerproduct_gemm_mem.ini'
    )

    if os.path.exists(value_path):
        os.remove(value_path)
        print(f"Deleted: {value_path}")
    else:
        print(f"File does not exist: {value_path}")

    dram_a_address, dram_b_address, dram_c_address = generate_outer_product_matrix(a, b, M, K, N)
    mem_init = os.path.join(extension_config.CONFIG_TORCHSIM_DIR, 'PyTorchSimBackend/extern/stonneCore/tests/outerproduct/outerproduct_gemm_mem.ini')
    a_row_init = os.path.join(extension_config.CONFIG_TORCHSIM_DIR, 'PyTorchSimBackend/extern/stonneCore/tests/outerproduct/outerproduct_gemm_rowpointerA.in')
    a_col_init = os.path.join(extension_config.CONFIG_TORCHSIM_DIR, 'PyTorchSimBackend/extern/stonneCore/tests/outerproduct/outerproduct_gemm_colpointerA.in')
    b_row_init = os.path.join(extension_config.CONFIG_TORCHSIM_DIR, 'PyTorchSimBackend/extern/stonneCore/tests/outerproduct/outerproduct_gemm_rowpointerB.in')
    b_col_init = os.path.join(extension_config.CONFIG_TORCHSIM_DIR, 'PyTorchSimBackend/extern/stonneCore/tests/outerproduct/outerproduct_gemm_colpointerB.in')
    c_result = os.path.join(extension_config.CONFIG_TORCHSIM_DIR, 'PyTorchSimBackend/extern/stonneCore/tests/outerproduct/result.out')
    graph = {
        0: {
            "node_id": 0,
            "node_name": "root",
            "node_type": 0,
            "parents": [],
            "children": [1]
        },
        1: {
            "node_id": 1,
            "node_name": "loopNode",
            "node_type": 2,
            "parents": [0],
            "children": [2],
            "loop_index": "loop_arg000",
            "loop_start": 0,
            "loop_end": 1,
            "loop_step": 1,
            "loop_type": "outer_loop"
        },
        2: {
            "node_id": 2,
            "node_name": "stonneNode",
            "node_type": 5,
            "parents": [1],
            "children": [],
            # Operation Type
            "stonne_operation": "outerProductGEMM",

            # GEMM Parameters
            "stonne_GEMM_K": K,
            "stonne_GEMM_N": N,
            "stonne_GEMM_M": M,
            "stonne_GEMM_T_K": 4,	# Currently fixed
            "stonne_GEMM_T_N": 1,	# Currently fixed
            "stonne_GEMM_T_M": 1,

            # Memory Initialization & File Paths
            "stonne_mem_init": os.path.join(extension_config.CONFIG_TORCHSIM_DIR, 'PyTorchSimBackend/extern/stonneCore/tests/outerproduct/outerproduct_gemm_mem.ini'),
            "stonne_mem_matrix_c_file_name": c_result,

            # Memory Addresses
            "stonne_matrix_a_dram_address": dram_a_address,
            "stonne_matrix_b_dram_address": dram_b_address,
            "stonne_matrix_c_dram_address": dram_c_address,

            # CSR & Bitmap Initialization
            "stonne_rowpointer_matrix_a_init": a_row_init,
            "stonne_colpointer_matrix_a_init": a_col_init,
            "stonne_rowpointer_matrix_b_init": b_row_init,
            "stonne_colpointer_matrix_b_init": b_col_init,
        }
    }
    source_code = "graph = " + str(graph)

    write_path = get_write_path(source_code)
    key, raw_tog_path = write(source_code, "py", specified_dir=write_path)
    tile_graph_generator = tog_generator(["flexagon_matmul"])
    tile_graph_generator.load_file(raw_tog_path)
    tile_graph_generator.generate_tile_graph(
        os.path.join(write_path, "tile_graph.onnx"),
        cycle_list=[0],
        offset=0,
        vector_lane=0
    )
    onnx_path = os.path.join(write_path, "tile_graph.onnx")
    #attribute_path = os.path.join(extension_config.CONFIG_TORCHSIM_DUMP_PATH, "tmp", hash_prefix(key), "attribute")
    backend_path = os.path.join(extension_config.CONFIG_TORCHSIM_DIR, "PyTorchSimBackend")
    stonne_config_path = f'{extension_config.CONFIG_TORCHSIM_DIR}/PyTorchSimBackend/configs/stonne_c1_simple_noc_tpuv3.json'
    backsim = BackendSimulator(backend_path, stonne_config_path)
    result_path = backsim.simulation(onnx_path)
    result = BackendSimulator.get_result_from_file(result_path)

    # Load result data
    with open(c_result, 'rb') as f:
        np_array = np.fromfile(f, dtype=TORCH_TO_NUMPY[out.dtype])
        src_tensor = torch.as_strided(torch.from_numpy(np_array), out.size(), out.stride())
        out.copy_(src_tensor.to(dtype=out.dtype))

custom_lib.define("_sparse_mm(Tensor a, Tensor b, Tensor out) -> Tensor")
custom_lib.impl("_sparse_mm", flexagon_frontend, "PrivateUse1")
custom_lib.impl("_sparse_mm", flexagon_frontend, "AutogradPrivateUse1")
