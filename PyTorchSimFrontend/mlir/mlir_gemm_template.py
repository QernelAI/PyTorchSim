import os
from typing import List, Optional, cast

from PyTorchSimFrontend.mlir.mlir_template import MLIRTemplate
from PyTorchSimFrontend.mlir.mlir_template import MLIRTemplateKernel
from torch._inductor.ir import Buffer
from torch._inductor.ir import IRNode
from torch._inductor.ir import ReinterpretView
from torch._inductor.codecache import write_atomic
import extension_codecache

GEMM_TEMPLATE = r"""
#map0 = affine_map<(d0, d1) -> (d0 * {{ K }} + d1)>
#map1 = affine_map<(d0, d1) -> (d0 * {{ N }} + d1)>
memref.global @X_spad : memref<{{ TILE_M }}x{{ TILE_K }}xf32, 1>
memref.global @W_spad : memref<{{ TILE_K }}x{{ TILE_N }}xf32, 1>
memref.global @Y_spad : memref<{{ TILE_M }}x{{ TILE_N }}xf32, 1>

func.func @{{ KERNEL_NAME }}{{kernel.def_kernel(inputs=[X, W, Bias], outputs=[Y], names_str="X, W, Bias, Y", input_reorder=input_reorder)}} {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c{{ TILE_K * 2 + 0}} = arith.constant {{ TILE_K * 2 + 0}} : index
  %M = arith.constant {{ M }} : index
  %N = arith.constant {{ N }} : index
  %K = arith.constant {{ K }} : index
  %X_buffer = memref.get_global @X_spad : memref<{{ TILE_M }}x{{ TILE_K }}xf32, 1>
  %W_buffer = memref.get_global @W_spad : memref<{{ TILE_K }}x{{ TILE_N }}xf32, 1>
  %Y_buffer = memref.get_global @Y_spad : memref<{{ TILE_M }}x{{ TILE_N }}xf32, 1>
  %tag = memref.alloc() : memref<1xi32>
  %v0 = arith.constant dense<0.0> : vector<{{ TILE_N }}xf32>

  affine.for %t_m = 0 to %M step {{ TILE_M }} {
    affine.for %t_n = 0 to %N step {{ TILE_N }} {
      affine.for %i = 0 to {{ TILE_M }} {
        affine.vector_store %v0, %Y_buffer[%i, 0] : memref<{{ TILE_M }}x{{ TILE_N }}xf32, 1>, vector<{{ TILE_N }}xf32>
      }
      affine.for %t_k = 0 to %K step {{ TILE_K }} {
        %index0 = affine.apply #map0(%t_m, %t_k)
        %index1 = affine.apply #map1(%t_k, %t_n)
        %index2 = affine.apply #map1(%t_m, %t_n)
        affine.dma_start %X[%index0], %X_buffer[0, 0], %tag[0], %c2, %K, %c2 : memref<{{ M * K }}xf32>, memref<{{ TILE_M }}x{{ TILE_K }}xf32, 1>, memref<1xi32>
        affine.dma_start %W[%index1], %W_buffer[0, 0], %tag[0], %c1, %N, %c2 : memref<{{ K * N }}xf32>, memref<{{ TILE_K }}x{{ TILE_N }}xf32, 1>, memref<1xi32>
        linalg.matmul ins(%X_buffer, %W_buffer : memref<{{ TILE_M }}x{{ TILE_K }}x{{ DATA_STYPE }}, 1>, memref<{{ TILE_K }}x{{ TILE_N }}x{{ DATA_STYPE }}, 1>)
                outs(%Y_buffer : memref<{{ TILE_M }}x{{ TILE_N }}x{{ DATA_STYPE }}, 1>)
        affine.dma_start %Y_buffer[0, 0], %Y[%index2], %tag[0], %c3, %N, %c2 : memref<{{ TILE_M }}x{{ TILE_N }}xf32, 1>, memref<{{ M * N }}xf32>, memref<1xi32>
      }
    }
  }
  return
}
"""

class MLIRGemmTemplate(MLIRTemplate):
    def __init__(self, input_nodes, layout, input_reorder=None):
        super().__init__("kernel", input_nodes, layout, input_reorder)

    def is_transposed(self, node):
        if isinstance(node, ReinterpretView):
            if node.layout.stride != node.data.layout.stride:
                if node.layout.stride[-2] == node.data.layout.stride[-1] and node.layout.stride[-1] == node.data.layout.stride[-2]:
                    return True
                else:
                  raise NotImplementedError("If the stride is not equal to the original stride, it should have been transposed.")
        return False

    def render(self,
               kernel: MLIRTemplateKernel,
               template_buffer_node = None,
               epilogue_nodes: Optional[List[IRNode]] = None,
               **kwargs):
        if template_buffer_node is not None:
            self.output_node = template_buffer_node
        if epilogue_nodes is not None and len(epilogue_nodes) > 0:
            self.output_node = cast(Buffer, epilogue_nodes[-1])

        X, W = self.input_nodes[0], self.input_nodes[1]
        Y = self.output_node
        Bias = None if len(self.input_nodes) == 2 else self.input_nodes[2]

        # Use BaseMLIRHardwareInfo
        TILE_M = min(kernel.vector_lane, X.get_size()[0])
        TILE_N = min(kernel.vector_lane, W.get_size()[1])
        TILE_K = min(kernel.vector_lane, X.get_size()[1])

        W_transposed = self.is_transposed(W)
        X_transposed = self.is_transposed(X)

        options = dict(
            KERNEL_NAME=self.name,
            kernel=kernel,
            M=X.get_size()[0],
            N=W.get_size()[1],
            K=X.get_size()[1],
            TILE_M=TILE_M,
            TILE_N=TILE_N,
            TILE_K=TILE_K,
            DATA_STYPE="f32",
            DATA_SIZE=4,
            X = X,
            W = W,
            Y = Y,
            Bias = Bias,
            W_transposed = W_transposed,
            X_transposed = X_transposed,
            input_reorder = self.input_reorder
        )
        code = self._template_from_string(GEMM_TEMPLATE).render(**options)
        write_path = extension_codecache.get_write_path(code)
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        write_path = os.path.join(write_path, "global_var.h")
        header = f"float X_spad[{TILE_M}][{TILE_K}] __attribute__ ((section(\".spad\")));\n"
        header += f"float W_spad[{TILE_K}][{TILE_N}] __attribute__ ((section(\".spad\")));\n"
        header += f"float Y_spad[{TILE_M}][{TILE_N}] __attribute__ ((section(\".spad\")));\n"
        if not os.path.exists(write_path):
            write_atomic(write_path, header)
        kernel.add_loop_info([options["M"], options["N"], options["K"]], [options["TILE_M"], options["TILE_N"], options["TILE_K"]])
        return code