import os
from typing import List, Optional, cast

from PyTorchSimFrontend.mlir.mlir_template import MLIRTemplate
from PyTorchSimFrontend.mlir.mlir_template import MLIRTemplateKernel
from torch._inductor.ir import Buffer
from torch._inductor.ir import IRNode
from torch._inductor.ir import ReinterpretView
from torch._inductor.codecache import write_atomic
import extension_codecache

# This template only represents the DMA operations
TEMPLATE = r"""#map0 = affine_map<(d0, d1, d2, d3) -> (d0 * {{ C * H * W }} + d1 * {{ H * W }} + d2 * {{ W }} + d3)>
memref.global @X_spad : memref<{{ in_tile }}x{{ in_tile }}xf32, 1>
memref.global @Y_spad : memref<{{ out_tile }}x{{ out_tile }}xf32, 1>

func.func @{{ KERNEL_NAME }} {{kernel.def_kernel(inputs=[X], outputs=[Y], names_str="X, Y")}} {
  %c_mvin = arith.constant 2 : index
  %c_mvout = arith.constant 3 : index
  %dummy = arith.constant 2 : index
  %B = arith.constant {{ B }} : index
  %C = arith.constant {{ C }} : index
  %H = arith.constant {{ H }} : index
  %W = arith.constant {{ W }} : index
  %in_chunk = arith.constant {{ in_tile * 2}} : index
  %out_chunk = arith.constant {{ out_tile * 2}} : index
  %X_buffer = memref.get_global @X_spad : memref<{{ in_tile }}x{{ in_tile }}xf32, 1>
  %Y_buffer = memref.get_global @Y_spad : memref<{{ out_tile }}x{{ out_tile }}xf32, 1>
  %tag = memref.alloc() : memref<1xi32>
  affine.for %b = 0 to %B {
    affine.for %c = 0 to %C {
      affine.for %h = 0 to %H step {{ out_tile }} {
        affine.for %w = 0 to %W step {{ out_tile }} {
          %index0 = affine.apply #map0(%b, %c, %h, %w)
          affine.dma_start %X[%index0], %X_buffer[0, 0], %tag[0], %c_mvin, %dummy, %in_chunk : memref<{{ IN }}xf32>, memref<{{ in_tile }}x{{ in_tile }}xf32, 1>, memref<1xi32>
          affine.dma_start %Y_buffer[0, 0], %Y[%index0], %tag[0], %c_mvout, %dummy, %out_chunk : memref<{{ out_tile }}x{{ out_tile }}xf32, 1>, memref<{{ OUT }}xf32>, memref<1xi32>
        } { outer_loop=true }
      } { outer_loop=true }
    } { outer_loop=true }
  } { outer_loop=true }
  return
}
"""

class MLIRMaxPoolTemplate(MLIRTemplate):
    def __init__(self, input_nodes, layout, kernel_size, stride, padding, dilation, ceil_mode):
        super().__init__("kernel", input_nodes, layout)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode

    def render(self,
               kernel: MLIRTemplateKernel,
               template_buffer_node = None,
               epilogue_nodes: Optional[List[IRNode]] = None,
               **kwargs):
        if template_buffer_node is not None:
            self.output_node = template_buffer_node
        if epilogue_nodes is not None and len(epilogue_nodes) > 0:
            self.output_node = cast(Buffer, epilogue_nodes[-1])
        X = self.input_nodes[0]
        Y = self.output_node
        out_tile = kernel.vector_lane
        in_tile = self.stride[0] * (out_tile - 1) + self.dilation[0] * (self.kernel_size[0] - 1) + 1 # padding should be considered? - 2 * self.padding
        options = {
          "KERNEL_NAME" : self.name,
          "kernel" : kernel,
          "IN" : X.get_numel(),
          "OUT" : Y.get_numel(),
          "X" : X,
          "Y" : Y,
          "B" : Y.get_size()[0],
          "C" : Y.get_size()[1],
          "H" : Y.get_size()[2],
          "W" : Y.get_size()[3],
          "in_tile" : in_tile,
          "out_tile" : out_tile,
        }
        code = self._template_from_string(TEMPLATE).render(**options)
        write_path = extension_codecache.get_write_path(code)
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        write_path = os.path.join(write_path, "global_var.h")
        header = f"float X_spad[{in_tile}][{in_tile}] __attribute__ ((section(\".spad\")));\n"
        header += f"float Y_spad[{out_tile}][{out_tile}] __attribute__ ((section(\".spad\")));\n"
        if not os.path.exists(write_path):
            write_atomic(write_path, header)
        kernel.add_loop_info([options["IN"]], [kernel.vector_lane, kernel.vector_lane])
        return code