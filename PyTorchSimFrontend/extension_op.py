import os
import subprocess
import math
import struct
import torch
from torch._inductor.select_algorithm import ExternKernelChoice

from PyTorchSimFrontend import extension_config

class MLIRExternKernelChoice(ExternKernelChoice):
    def call_name(self):
        return f"torch.ops.extension_op.{self.name}"

custom_lib = torch.library.Library("extension_op", "DEF")

def _sparse_mm(a, b, out):
    print("PYTHON CUSTOM OP EXAMPLE")
    out.copy_(a + b)

def generate_outer_product_matrix(a, outer, inner, name):
	a_cpu = a.cpu()
	value_pointer = os.path.join(extension_config.CONFIG_TORCHSIM_DIR,
		'PyTorchSimBackend/extern/stonneCore/tests/outerproduct/outerproduct_gemm_mem.ini')
	row_pointer = os.path.join(extension_config.CONFIG_TORCHSIM_DIR,
		f'PyTorchSimBackend/extern/stonneCore/tests/outerproduct/outerproduct_gemm_rowpointer{name}.in')
	col_pointer = os.path.join(extension_config.CONFIG_TORCHSIM_DIR,
		f'PyTorchSimBackend/extern/stonneCore/tests/outerproduct/outerproduct_gemm_colpointer{name}.in')

	with open(value_pointer, "a") as fd, open(row_pointer, "w") as rp, open(col_pointer, "w") as cp:
    #generating matrix
		n_nonzeros=0
		for o in range(outer):  # col major
			rp.write(str(n_nonzeros)+","); # writing the index
			for i in range(inner):
				value = a_cpu[i, o]
				if value:  # value is generated
					if((i==(inner-1)) and (o==(outer-1))):
						cp.write(str(i))
					else:
						cp.write(str(i)+","); #writing the row index
					ba = bytearray(struct.pack(">f", value))  # generating list of bytes
					my_int = int.from_bytes(ba, "big")
					fd.write(str(my_int))
					fd.write(",")
					n_nonzeros+=1

def flexagon_frontend(a, b, out):
	print("FLEXAGON FRONTEND")
	x_shape = a.shape
	w_shape = b.shape

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

	generate_outer_product_matrix(a, K, M, "A")
	generate_outer_product_matrix(b, K, N, "B")


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
			"stonne_GEMM_T_M": 1,  # 기본값 설정 (T_M이 빠져있으므로 1로 설정)

			# Memory Initialization & File Paths
			"stonne_mem_init": os.path.join(extension_config.CONFIG_TORCHSIM_DIR, 'PyTorchSimBackend/extern/stonneCore/tests/outerproduct/outerproduct_gemm_mem.ini'),
			"stonne_mem_matrix_c_file_name": os.path.join(extension_config.CONFIG_TORCHSIM_DIR, 'PyTorchSimBackend/extern/stonneCore/tests/outerproduct/result.out'),

			# Memory Addresses
			"stonne_matrix_a_dram_address": 0,
			"stonne_matrix_b_dram_address": 12444,
			"stonne_matrix_c_dram_address": 24608,

			# CSR & Bitmap Initialization
			"stonne_rowpointer_matrix_a_init": os.path.join(extension_config.CONFIG_TORCHSIM_DIR, 'PyTorchSimBackend/extern/stonneCore/tests/outerproduct/outerproduct_gemm_rowpointerA.in'),
			"stonne_colpointer_matrix_a_init": os.path.join(extension_config.CONFIG_TORCHSIM_DIR, 'PyTorchSimBackend/extern/stonneCore/tests/outerproduct/outerproduct_gemm_colpointerA.in'),
			"stonne_rowpointer_matrix_b_init": os.path.join(extension_config.CONFIG_TORCHSIM_DIR, 'PyTorchSimBackend/extern/stonneCore/tests/outerproduct/outerproduct_gemm_rowpointerB.in'),
			"stonne_colpointer_matrix_b_init": os.path.join(extension_config.CONFIG_TORCHSIM_DIR, 'PyTorchSimBackend/extern/stonneCore/tests/outerproduct/outerproduct_gemm_colpointerB.in'),
		}
	}

	out.copy_(a + b)

custom_lib.define("_sparse_mm(Tensor a, Tensor b, Tensor out) -> Tensor")
custom_lib.impl("_sparse_mm", flexagon_frontend, "PrivateUse1")
custom_lib.impl("_sparse_mm", flexagon_frontend, "AutogradPrivateUse1")
