import dataclasses
import contextlib
import sympy
import itertools
import re
import os
from functools import reduce
from operator import mul
from typing import List
from typing import Dict
import torch
from torch._inductor import dependencies
from torch._inductor.codegen import cpp, wrapper, common
from torch._inductor.scheduler import BaseScheduling
from torch._inductor.virtualized import V, _ops as ops
from torch._inductor.codecache import write_atomic, write
from Simulator.simulator import BackendSimulator
from torch._inductor.utils import (
    IndentedBuffer,
    is_welford_reduction,
)
import extension_codecache


from . import mlir_common
from . import mlir_lowering

def reduction_init(reduction_type, dtype):
    if dtype in cpp.DTYPE_LOWP_FP:
        # Since load promotes all half-precision inputs to float, the initial
        # constant for reduction must be promoted as well
        dtype = torch.float32
    if reduction_type in ("xor_sum", "sum", "any"):
        return "0.0"
    if reduction_type == "prod":
        return "1.0"
    if reduction_type in {"max", "argmax"}:
        return "0.0"
    if reduction_type in {"min", "argmin"}:
        return "0.0"
    if reduction_type in {"welford_reduce"}:
        return f"0.0"
    raise AssertionError(reduction_type)

def reduction_combine(reduction_type, start_value, vector_value, tile_size=64):
    if reduction_type == "sum":
        return f"arith.addf %{start_value}, %{vector_value} : vector<{tile_size}xf32>"
    if reduction_type == "prod":
        return f"arith.mulf %{start_value}, %{vector_value} : vector<{tile_size}xf32>"
    if reduction_type == "xor_sum":
        raise NotImplementedError() # TODO: implement
    if reduction_type == "any":
        raise NotImplementedError()
    if reduction_type in ("min", "max"):
        raise NotImplementedError()
    if reduction_type == "welford_reduce":
        raise NotImplementedError()
    if reduction_type == "welford_combine":
        raise NotImplementedError()
    raise AssertionError(reduction_type)

def reduction_combine_vec(reduction_type, vector_value, init_value, axis, shape, reduced_shape):
    if reduction_type == "sum":
        return f"vector.multi_reduction <add>, %{vector_value}, %{init_value} [{axis}] : {shape} to {reduced_shape}"
    if reduction_type == "prod":
        return f"vector.multi_reduction <mul>, %{vector_value}, %{init_value} [{axis}] : {shape} to {reduced_shape}"
    if reduction_type == "max":
        return f"vector.multi_reduction <maximumf>, %{vector_value}, %{init_value} [{axis}] : {shape} to {reduced_shape}"
    if reduction_type == "min":
        return f"vector.multi_reduction <minimumf>, %{vector_value}, %{init_value} [{axis}] : {shape} to {reduced_shape}"
    if reduction_type == "any":
        return f"vector.multi_reduction <and>, %{vector_value}, %{init_value} [{axis}] : {shape} to {reduced_shape}"
    raise AssertionError(reduction_type)

class ExtensionWrapperCodegen(wrapper.WrapperCodeGen):
    def __init__(self):
        super().__init__()

    def write_header(self):
        self.header.splice(
            f"""
                from ctypes import c_void_p, c_long
                import torch
                import math
                import random
                import os
                import tempfile
                from math import inf, nan
                from torch._inductor.hooks import run_intermediate_hooks
                from torch._inductor.utils import maybe_profile
                from torch._inductor.codegen.memory_planning import _align as align

                from torch import device, empty, empty_strided
                from {extension_codecache.__name__} import CustomAsyncCompile
                from torch._inductor.select_algorithm import extern_kernels

                aten = torch.ops.aten
                inductor_ops = torch.ops.inductor
                assert_size_stride = torch._C._dynamo.guards.assert_size_stride
                alloc_from_pool = torch.ops.inductor._alloc_from_pool
                reinterpret_tensor = torch.ops.aten._reinterpret_tensor
                async_compile = CustomAsyncCompile()
                os.environ["TORCHSIM_LAST_COMPILED_MODULE"] = __file__
            """
        )

class ExtensionOverrides(common.OpOverrides):
    @staticmethod
    def add(operand1, operand2, tile_size=16, dtype="f32"):
        shape = f"vector<{tile_size}x{dtype}>" if tile_size > 1 else dtype
        return f'arith.add{dtype[0]} %{operand1}, %{operand2} : {shape}'

    @staticmethod
    def sub(operand1, operand2, tile_size=16, dtype="f32"):
        shape = f"vector<{tile_size}x{dtype}>" if tile_size > 1 else dtype
        return f'arith.sub{dtype[0]} %{operand1}, %{operand2} : {shape}'

    @staticmethod
    def mul(operand1, operand2, tile_size=16, dtype="f32"):
        shape = f"vector<{tile_size}x{dtype}>" if tile_size > 1 else dtype
        return f'arith.mul{dtype[0]} %{operand1}, %{operand2} : {shape}'

    @staticmethod
    def div(operand1, operand2, tile_size=16, dtype="f32"):
        shape = f"vector<{tile_size}x{dtype}>" if tile_size > 1 else dtype
        return f'arith.div{dtype[0]} %{operand1}, %{operand2} : {shape}'

    @staticmethod
    def truediv(operand1, operand2, tile_size=16, dtype="f32"):
        shape = f"vector<{tile_size}x{dtype}>" if tile_size > 1 else dtype
        return f'arith.div{dtype[0]} %{operand1}, %{operand2} : {shape}'

    @staticmethod
    def to_dtype(x, dtype, src_dtype=None, tile_size=16):
        mlir_dtype = mlir_common.DTYPE_TO_MLIR[dtype]
        src_mlir_dtype = mlir_common.DTYPE_TO_MLIR[src_dtype]

        dst_bits = 1 if dtype == torch.bool else torch.finfo(dtype).bits if dtype.is_floating_point else torch.iinfo(dtype).bits
        src_bits = 1 if src_dtype == torch.bool else torch.finfo(src_dtype).bits if src_dtype.is_floating_point else torch.iinfo(src_dtype).bits
        shape = f"vector<{tile_size}x{mlir_dtype}>" if tile_size > 1 else mlir_dtype
        src_shape = f"vector<{tile_size}x{src_mlir_dtype}>" if tile_size > 1 else src_mlir_dtype
        if dtype.is_floating_point and not src_dtype.is_floating_point:
            return f"arith.sitofp %{x} : {src_shape} to {shape}"
        elif not dtype.is_floating_point and src_dtype.is_floating_point:
            return f"arith.fptosi %{x} : {src_shape} to {shape}"
        else:
            operation = "arith.trunc" if dst_bits < src_bits else "arith.ext"
            operation_suffix = "f" if dtype.is_floating_point else "i"
            return f"{operation}{operation_suffix} %{x} : {src_shape} to {shape}"

    @staticmethod
    def constant(value, src_type, tile_size=16, dtype="f32"):
        src_type = mlir_common.DTYPE_TO_MLIR[src_type]
        # if value represented by e notation, convert to float (ex 1e-3 -> 1.0e-3)
        if "e" in str(value):
            value = float(value)
        if src_type[0] == "f":
            value = format(value, ".20f")
        return f'arith.constant {value} : {src_type}'

    @staticmethod
    def exp(operand, tile_size=16, dtype="f32"):
        shape = f"vector<{tile_size}x{dtype}>" if tile_size > 1 else dtype
        return f'math.exp %{operand} : {shape}'

    @staticmethod
    def maximum(operand1, operand2, tile_size=16, dtype="f32"):
        shape = f"vector<{tile_size}x{dtype}>" if tile_size > 1 else dtype
        return f'arith.maximum{dtype[0]} %{operand1}, %{operand2} : {shape}'

    @staticmethod
    def sqrt(x, tile_size=16, dtype="f32"):
        shape = f"vector<{tile_size}x{dtype}>" if tile_size > 1 else dtype
        return f'math.sqrt %{x} : {shape}'

    @staticmethod
    def ne(operand1, operand2, tile_size=16, dtype="f32"):
        shape = f"vector<{tile_size}xi1>" if tile_size > 1 else "i1"
        return f'arith.cmpi one, %{operand1}, %{operand2} : {shape}'

    @staticmethod
    def le(operand1, operand2, tile_size=16, dtype="f32"):
        shape = f"vector<{tile_size}x{dtype}>" if tile_size > 1 else dtype
        return f'arith.cmp{dtype[0]} ole, %{operand1}, %{operand2} : {shape}'

    @staticmethod
    def relu(x, tile_size=16, dtype=None):
        return ops.maximum(x, ops.constant(0.0, torch.float32))

    @staticmethod
    def sigmoid(x, tile_size=16, dtype=None):
        one = ops.constant(1, torch.float32)
        return ops.truediv(one, ops.add(one, ops.exp(ops.neg(x))))

    @staticmethod
    def neg(x, tile_size=16, dtype="f32"):
        shape = f"vector<{tile_size}x{dtype}>" if tile_size > 1 else dtype
        return f'arith.neg{dtype[0]} %{x} : {shape}'

    @staticmethod
    def where(condition, x, y, tile_size=16, dtype="f32"):
        shape = f"vector<{tile_size}x{dtype}>" if tile_size > 1 else dtype
        cond_shape = f"vector<{tile_size}xi1>," if tile_size > 1 else ""
        return f"arith.select %{condition}, %{x}, %{y} : {cond_shape} {shape}"

    @staticmethod
    def logical_not(operand, tile_size=16, dtype="f32"):
        tile_size=16
        shape = f"vector<{tile_size}x{dtype}>" if tile_size > 1 else dtype
        result_shape = f"vector<{tile_size}xi1>" if tile_size > 1 else "i1"
        raise NotImplementedError("logical_not")
        return f"arith.cmp{dtype[0]} oeq, %{operand}, %zero_vec{tile_size} : {shape} -> {result_shape}"

    @staticmethod
    def rsqrt(x, tile_size=16, dtype="f32"):
        shape = f"vector<{tile_size}x{dtype}>" if tile_size > 1 else dtype
        return f'math.rsqrt %{x} : {shape}'

    @staticmethod
    def pow(a, b, tile_size=16, dtype="f32"):
        shape = f"vector<{tile_size}x{dtype}>" if tile_size > 1 else dtype
        return f"math.pow{dtype[0]} %{a}, %{b} : {shape}"

    @staticmethod
    def log(x, tile_size=16, dtype="f32"):
        shape = f"vector<{tile_size}x{dtype}>" if tile_size > 1 else dtype
        return f'math.log %{x} : {shape}'

    @staticmethod
    def reciprocal(a, tile_size=16, dtype="f32"):
        return ops.div(ops.constant(1.0, torch.float32), a)

RTYPE_TO_MLIR = {
    "sum": "add",
    "prod": "mul",
}

DMA_TYPE = {
    "MVIN1": 2,
    "MVIN2": 1,
    "MVIN3": 14,
    "MVOUT": 3,
}

class MLIRTile():
    def __init__(self, n_row, n_col, vector_lane) -> None:
        self.n_row = n_row
        self.n_col = n_col
        self.vector_lane = vector_lane
        self.axis_strides= {}

    def get_tile_size(self):
        return self.n_row * self.n_col

    def get_rows_per_lane(self):
        if self.n_row % self.vector_lane != 0:
            print("[Warning] n_row % vector_lane != 0")
        return self.n_row // self.vector_lane

    def get_cols_per_lane(self):
        if self.n_col % self.vector_lane != 0:
            print("[Warning] n_col % vector_lane != 0")
        return self.n_col // self.vector_lane

    def get_tile_size_per_lane(self):
        if self.get_tile_size() % self.vector_lane != 0:
            print("[Warning] n_col % vector_lane != 0")
        return self.get_tile_size() // self.vector_lane

    def get_tile_shape(self):
        return f"{self.n_row}x{self.n_col}"

    def get_chunk_size(self, is_vector_lane_row_major):
        if self.is_vector():
            return self.get_tile_size_per_lane()
        if is_vector_lane_row_major:
            chunk_size = self.get_tile_size_per_lane()
        else:
            chunk_size = self.get_cols_per_lane()
        return chunk_size

    def is_vector(self):
        return self.n_row == 1

    def update_axis_stride(self, cv):
        if any([i==0 for i in cv]):
            return

        for axis, stride in enumerate(cv.values()):
            self.axis_strides[axis] = stride

    def get_axis_and_tile_info(self):
        axis = list(self.axis_strides.keys())
        return {axis[-2]: self.n_row, axis[-1]: self.n_col}

class MLIRKernel(mlir_common.BaseMLIRKernel):
    overrides = ExtensionOverrides
    newvar_prefix = "%"

    def __init__(self, args=None):
        super().__init__(mlir_common.MLIRKernelArgs())
        self.call_ranges = None
        self.ranges = None
        self.itervars = None
        self.reduction_depth = None
        self.reduction_prefix = IndentedBuffer()
        self.reduction_suffix = IndentedBuffer()
        self.global_vars = IndentedBuffer()
        self.global_vars_set = set()
        self.header = IndentedBuffer()
        self.reduction_vars = {}
        self.reduction_cse = common.CSE(self.newvar_prefix, self.suffix, name_prefix="tmp_acc")
        self.iterator_cse = common.CSE(self.newvar_prefix, self.suffix, name_prefix="iter")
        self.init_cse = common.CSE(self.newvar_prefix, self.suffix, name_prefix="init")
        self.init_vec_cse = common.CSE(self.newvar_prefix, self.suffix, name_prefix="init_vec")
        self.map_cse = common.CSE("#", self.suffix, name_prefix="map")
        self.consts = set()
        self.tags = set()
        self.tile_desc = MLIRTile(self.tile_row, self.tile_col, self.vector_lane)
        self.dma_cache = {}
        self.dma_counter = 1
        self.reduction_idx = {}
        self.affine_yield = {}
        self.welford_reduce_out = None
        self.reduce_iterator = {}

<<<<<<< HEAD
        # is_transposed
        if len(V.graph.graph_inputs) == 1 and len(V.graph.buffers):
            input_shape = list(V.graph.graph_inputs.values())[0].layout.size
            output_shape = V.graph.buffers[0].layout.size
            if len(input_shape) > 1 and len(output_shape) > 1 and input_shape[-1] == output_shape[-2] and input_shape[-2] == output_shape[-1]:
                self.is_transpose = True

=======
>>>>>>> eccd7c1 ([Frontend] Reduction mechanism reworking)
    def get_constant_vector(self, expr):
        constant_vector = [int(expr.coeff(var)) for var in self.itervars]
        return constant_vector

    def find_node_by_name(self, name):
        if name in V.graph.graph_inputs:
            return V.graph.graph_inputs[name]
        else:
            for output_node in V.graph.graph_outputs:
                if output_node.anme == name:
                    return output_node

    def get_dma_info(self, name, index, dtype, is_store):
        cv = self.get_constant_vector(index)
        tile_size_per_lane = min(self.tile_desc.get_tile_size_per_lane(), self.buffer_types[name][1])
        self.tile_desc.update_axis_stride(cv)

        # Case 0. Tile is 0-D scalar
        if len(cv) == 0:
            raise NotImplementedError()
            chunk_size = self.tile_desc.get_tile_size()
            is_col_major = False
            stride = self.tile_desc.get_tile_size()
            tile_size_per_lane = min(self.tile_desc.get_tile_size(), self.buffer_types[name][1])
        # Case 1. Tile is 1-D vector type
        elif len(cv) == 1 and len(cv) <= self.reduction_depth:
            is_vector_lane_row_major = False
            is_col_major = True # Actually it is not needed in vector case
            t_row = 1
            t_col = self.tile_desc.get_tile_size()
            chunk_size = self.tile_desc.get_tile_size_per_lane()
            mm_stride = t_col
        # Case 2. Tile is 1-D vector type with reduction
        elif len(cv) == 1 and len(cv) == self.reduction_depth + 1:
            # Use only one vectorlane to reduce a vector
            is_vector_lane_row_major = True
            is_col_major = False
            t_row = 1
            t_col = self.tile_desc.get_tile_size()
            chunk_size = self.tile_desc.get_tile_size()
        # Case 3. Tile is 2-D tile
        elif len(cv) == 2:
            if cv[0] != 0 and cv[1] != 0:
                is_reduction = self.reduction_depth == 1
                is_transposed = cv[0] < cv[1]
                if is_transposed:
                    t_row = self.tile_desc.n_col
                    t_col = self.tile_desc.n_row
                    mm_stride = self.ranges[0]
                else:
                    t_row = self.tile_desc.n_row
                    t_col = self.tile_desc.n_col
                    mm_stride = self.ranges[1]

                if is_reduction and is_transposed:
                    is_col_major = False
                    chunk_size = t_col // self.vector_lane
                elif is_reduction and not is_transposed:
                    is_col_major = True
                    chunk_size = self.tile_desc.get_tile_size() // self.vector_lane
                elif not is_reduction and is_transposed:
                    # Transposed case
                    is_col_major = True
                    chunk_size = self.tile_desc.get_tile_size() // self.vector_lane
                else:
                    is_col_major = False
                    chunk_size = self.tile_desc.get_cols_per_lane()
            else:
                # Broadcast pattern
                chunk_size = self.tile_desc.get_cols_per_lane()
                is_col_major = False
                mm_stride = 0
                if cv[1] == 0:
                    t_row = self.vector_lane
                    t_col = self.tile_desc.get_cols_per_lane()
                else: # cv[0] == 0
                    raise NotImplementedError()
                    t_row = self.vector_lane
                    t_col = self.tile_desc.get_cols_per_lane()
        else:
            raise NotImplementedError()

        assert(not (dtype==torch.bool and chunk_size < 8))
        chunk = chunk_size << 1 | is_col_major
        return mm_stride, chunk, [t_row, t_col], tile_size_per_lane

    def parse_indices(self, expr):
        if len(expr.args) == 0:
            return expr, expr
        for itervars in self.itervars:
            new_coeff = ((expr.coeff(itervars) + self.tile_desc.n_col - 1) // self.tile_desc.n_col) * self.tile_desc.n_col
            expr = expr.subs(expr.coeff(itervars), new_coeff)
        expr_str = str(expr)
        pattern = r'index\d+'
        indices = set(re.findall(pattern, expr_str))
        args = ", ".join(map(str, indices))
        if "//" in expr_str:
            expr_str = expr_str.replace("//", " floordiv ")
        pattern = r"ModularIndexing\((.*?)\)"
        matches = re.search(pattern, expr_str)
        if matches:
            mod_args = matches.group(1)
            args_list = mod_args.split(", ")
            replace_str = f"({args_list[0]} floordiv {args_list[1]}) mod {args_list[2]}"
            expr_str = re.sub(r"ModularIndexing\([^)]*\)", replace_str, expr_str)

        map_var = self.map_cse.generate(self.global_vars, f"affine_map<({args}) -> ({expr_str})>")
        args = ", ".join([f"%{i}" for i in indices])
        index = self.cse.generate(self.loads, f"affine.apply #{map_var}({args})")
        return index, expr

    def codegen_nodes(self, nodes, kernel_name):
        _, (group, reduction_group) = max(
            nodes, key=lambda x: int(x.is_reduction())
        ).group

        self.set_ranges(group, reduction_group)
        with self as kernel:
            for node in nodes:
                vars, reduction_vars = kernel.set_ranges(group, reduction_group)
                self.args = mlir_common.MLIRKernelArgs(self.tile_desc.n_row, self.tile_desc.n_col)
                _, _, _, self.buffer_types = self.args.mlir_argdefs()
                self.reduction_idx = {var: i for i, var in enumerate(reduction_vars)}
                node.run(vars, reduction_vars)

        src_code = self.codegen_kernel(kernel_name=kernel_name)
        self.meta_kernel()

        write_path = extension_codecache.get_write_path(src_code)
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        write_path = os.path.join(write_path, "global_var.h")
        if not os.path.exists(write_path):
            write_atomic(write_path, self.header.getvalue())
        return src_code

    def load(self, name: str, index: sympy.Expr):
        index = self.rename_indexing(index)
        indices, index = self.parse_indices(index)
        prefix = "" if index.is_number else "%"
        var = self.args.input(name)
        dtype = V.graph.get_dtype(name)
        type_name = mlir_common.DTYPE_TO_MLIR[dtype]
        stride, chunk, tile_shape, tile_size_per_lane = self.get_dma_info(name, index, dtype, 0)
        dram_tile_shape = f"{tile_shape[0]}x{tile_shape[1]}"

        # Define scratch pad buffer
        buffer = self.get_scratchpad_buffer(dtype, name, self.tile_desc.n_row, self.tile_desc.n_col, dram_tile_shape, self.loads)

        # MVIN Encoding
        dma_key = (stride, chunk, dtype)
        if dma_key in self.dma_cache:
            dmaType, stride, chunk = self.dma_cache[dma_key]
        else:
            assert(self.dma_counter < 4)
            dmaType = DMA_TYPE[f"MVIN{self.dma_counter}"]
            self.dma_counter += 1
            self.consts.add(dmaType)
            self.consts.add(stride)
            self.consts.add(chunk)
            self.dma_cache[dma_key] = dmaType, stride, chunk
        self.tags.add(f"{name}_tag")
        code = f"affine.dma_start %{var}[{prefix}{indices}], %{buffer}[0, 0], %{name}_tag[0], %c{dmaType}, %c{stride}, %c{chunk} : memref<{self.buffer_types[name][1]}x{type_name}>, memref<{dram_tile_shape}x{type_name}, 1>, memref<1xi32>"
        self.cse.generate(self.loads, code, assignment = False) # FIXME: assignment = False does not support caching

        operation = "affine.vector_load" if tile_size_per_lane > 1 else "affine.load"
        shape = f", vector<{tile_size_per_lane}x{type_name}>" if tile_size_per_lane > 1 else ""
        line = f"{operation} %{buffer}[0, 0] : memref<{dram_tile_shape}x{type_name}, 1>{shape}"
        out = self.cse.generate(self.loads, line)
        self.tile_info[out] = tile_size_per_lane, dtype
        return out

    def store(self, name: str, index: sympy.Expr, value, *args, **kwargs):
        index = self.rename_indexing(index)
        indices, index = self.parse_indices(index)
        prefix = "" if index.is_number else "%"
        var = self.args.output(name)
        dtype = V.graph.get_dtype(name)
        type_name = mlir_common.DTYPE_TO_MLIR[dtype]
        stride, chunk, tile_shape, tile_size_per_lane = self.get_dma_info(name, index, dtype, 1)
        dram_tile_shape = f"{tile_shape[0]}x{tile_shape[1]}"

        # Define scratch pad buffer
        buffer = self.get_scratchpad_buffer(dtype, name, self.tile_desc.n_row, self.tile_desc.n_col, dram_tile_shape, self.stores)

        # MVOUT Encoding
        dmaType = 3 # MVIN 2, MVIN2 1, MVIN3 14, MVOUT 3
        self.consts.add(dmaType)
        self.consts.add(stride)
        self.consts.add(chunk)

        operation = "affine.vector_store" if tile_size_per_lane > 1 else "affine.store"
        shape = f", vector<{tile_size_per_lane}x{type_name}>" if tile_size_per_lane > 1 else ""
        line = f"{operation} %{value}, %{buffer}[0, 0] : memref<{dram_tile_shape}x{type_name}, 1>{shape}"
        self.cse.generate(self.stores, line, assignment = False)
        self.tags.add(f"{name}_tag")
        code = f"affine.dma_start %{buffer}[0, 0], %{var}[{prefix}{indices}], %{name}_tag[0], %c{dmaType}, %c{stride}, %c{chunk} : memref<{dram_tile_shape}x{type_name}, 1>, memref<{self.buffer_types[name][1]}x{type_name}>, memref<1xi32>"
        self.cse.generate(self.stores, code, assignment = False)

    def reduction(self, dtype, src_dtype, reduction_type, value):
        argmax_or_argmin = reduction_type in {"argmax", "argmin"}
        if argmax_or_argmin:
            raise NotImplementedError() #TODO: argmin, argmax
        elif is_welford_reduction(reduction_type):
            if reduction_type == "welford_combine":
                raise NotImplementedError("welford_combine")
            else:
                assert reduction_type == "welford_reduce"
                type_name = mlir_common.DTYPE_TO_MLIR[dtype]
                reduction_key = src_dtype, reduction_type, value
                sum = self.reduction(dtype, src_dtype, "sum", value)
                sqr_sum = self.reduction(dtype, src_dtype, "sum", ops.mul(value, value))
                self.welford_reduce_out = (sum, sqr_sum, None)
                return sum, sqr_sum, None
        else:
            reduction_key = src_dtype, reduction_type, value
            acc = self.reduction_cse.generate(
                self.loads, f"reduction {reduction_key}", write=False
            )
            iterator = self.iterator_cse.generate(
                self.loads, f"reduction {reduction_key}", write=False
            )
            init = self.init_cse.generate(
                self.loads, f"reduction {reduction_key}", write=False
            )
            init_vec = self.init_vec_cse.generate(
                self.loads, f"reduction {reduction_key}", write=False
            )
            type_name = mlir_common.DTYPE_TO_MLIR[dtype]
            acc_var = init
            acc_shape = type_name
            shape = f"vector<{self.tile_desc.get_tile_size()}x{type_name}>"
            reduced_shape = type_name
            self.reduction_prefix.writeline(f"%{init} = arith.constant {reduction_init(reduction_type, dtype)} : {type_name}")
            if len(self.ranges) == 2:
                vec_len = self.tile_desc.n_row // self.tile_desc.get_rows_per_lane()
                flattened_size = f"vector<{self.tile_desc.get_tile_size_per_lane()}x{type_name}>"

                # It is column majored per lane tile
                expaned_size = f"vector<{self.tile_desc.get_tile_size_per_lane()//vec_len}x{vec_len}x{type_name}>"
                value = self.cse.generate(self.compute, f"vector.shape_cast %{value} : {flattened_size} to {expaned_size}")
                shape = expaned_size

                # Edge case for scalar
                if vec_len == 1:
                    reduced_shape = f"{type_name}"
                    init_vec = init
                    axis = "0, 1"
                    acc_var = init
                else:
                    reduced_shape = f"vector<{vec_len}x{type_name}>"
                    self.reduction_prefix.writeline(f"%{init_vec} = vector.broadcast %{init} : {type_name} to {reduced_shape}")
                    axis = "0"
                    acc_var = init_vec
            else:
                raise NotImplementedError()

            self.reduction_vars[acc] = (reduction_type, iterator, acc_var, reduced_shape)
            out = self.cse.generate(self.compute, reduction_combine_vec(reduction_type, value, iterator, axis=axis, shape=shape, reduced_shape=reduced_shape))
            self.affine_yield[out] = reduced_shape

            self.reduction_cse.reduction_cache[reduction_key] = acc
            self.iterator_cse.reduction_cache[reduction_key] = iterator
            self.init_cse.reduction_cache[reduction_key] = init_vec
        return acc

    def store_reduction(self, name, index, value):
        var = self.args.output(name)
        dtype = V.graph.get_dtype(name)
        type_name = mlir_common.DTYPE_TO_MLIR[dtype]
        index = self.rename_indexing(index)
        indices, index = self.parse_indices(index)
        prefix = "" if index.is_number else "%"

        # Tile is always reuduced in inner loop
        tile_col = self.tile_desc.n_row
        tile_row = 1
        dram_tile_shape = f"{tile_row}x{tile_col}"
        buffer = self.get_scratchpad_buffer(dtype, name, tile_row, tile_col, dram_tile_shape, self.reductions_suffix)
        if self.welford_reduce_out is not None:
            raise NotImplementedError()
            sum, sqr_sum, _ = self.welford_reduce_out
            shape = f"vector<{self.tile_col_per_lane}x{type_name}>" if self.buffer_types[name][1] > 1 else type_name
            # mean
            self.cse.generate(self.reductions_suffix, f"%f{self.ranges[self.reduction_depth]} = arith.constant {float(self.ranges[self.reduction_depth])} : f32", assignment=False)
            if self.buffer_types[name][1] > 1:
                divider_vec = self.cse.generate(self.reductions_suffix, f"vector.broadcast %f{self.ranges[self.reduction_depth]} : f32 to vector<{self.tile_col_per_lane}x{type_name}>")
            else:
                divider_vec = f"f{self.buffer_types[name][1]}"
            mean = self.cse.generate(self.reductions_suffix, f"arith.divf %{sum}, %{divider_vec} : {shape}")

            # m2 = (E(X^2) - E(X)^2) * N
            sqr_mean = self.cse.generate(self.reductions_suffix, f"arith.divf %{sqr_sum}, %{divider_vec} : {shape}")
            mean_sqr = self.cse.generate(self.reductions_suffix, f"arith.mulf %{mean}, %{mean} : {shape}")
            variance = self.cse.generate(self.reductions_suffix, f"arith.subf %{sqr_mean}, %{mean_sqr} : {shape}")
            m2 = self.cse.generate(self.reductions_suffix, f"arith.mulf %{variance}, %{divider_vec} : {shape}")
            if name == "buf0": # FIXME: check which is mean or m2
                value = mean
            else:
                value = m2

        # Select mlir store operaiton
        if self.buffer_types[name][1] == 1 or self.tile_desc.get_rows_per_lane() == 1:
            operation = "affine.store"
            raise NotImplementedError("Scalar store!")
        else:
            operation =  "affine.vector_store"

        # Select src type
        if self.tile_desc.get_rows_per_lane() == 1:
            shape = ""
        else:
            shape = f"vector<{self.tile_desc.get_rows_per_lane()}x{type_name}>"
            shape = f", {shape}" if self.buffer_types[name][1] > 1 else ""
        line = f"{operation} %{value}, %{buffer}[0, 0] : memref<{tile_row}x{tile_col}x{type_name}, 1>{shape}"
        self.cse.generate(self.reductions_suffix, line, assignment = False)

        # MVOUT Encoding
        dmaType = 3 # MVIN 2, MVIN2 1, MVIN3 14, MVOUT 3
        mm_stride = tile_col
        is_col_major = False
        chunk_size = self.tile_desc.get_rows_per_lane()
        chunk = chunk_size << 1 | is_col_major
        self.consts.add(dmaType)
        self.consts.add(mm_stride)
        self.consts.add(chunk)
        self.tags.add(f"{name}_tag")
        # Change row, col
        code = f"affine.dma_start %{buffer}[0, 0], %{var}[{prefix}{indices}], %{name}_tag[0], %c{dmaType}, %c{mm_stride}, %c{chunk} : memref<{tile_row}x{tile_col}x{type_name}, 1>, memref<{self.buffer_types[name][1]}x{type_name}>, memref<1xi32>"
        self.cse.generate(self.reductions_suffix, code, assignment = False)

    def codegen_init(self):
        code = IndentedBuffer()
        tags = sorted(self.tags)
        consts = sorted(self.consts)
        for tag in tags:
            code.writeline(f"%{tag} = memref.alloc() : memref<1xi32>")
        for const in consts:
            code.writeline(f"%c{const} = arith.constant {const} : index")
        return code

    def codegen_loops(self):
        code = mlir_common.ParallelLoopBuffer()
        # Loop body part
        tile_row, tile_col = self.tile_desc.n_row, self.tile_desc.n_col
        # FIXME.
        #if (self.tiling_idx < self.reduction_depth and len(self.reduction_idx) > 0):
        #    tile_row, tile_col = self.tile_desc.n_col, self.tile_desc.n_row
        tile_row = self.tile_desc.get_tile_size() if len(self.itervars) == 1 else tile_row
        loops = [LoopLevel(var, size, idx, tile_row=tile_row, tile_col=tile_col) for idx, (var, size) in enumerate(zip(self.itervars, self.ranges))]
        loops, reductions = [LoopNest(loops[: self.reduction_depth]),
                             LoopNest(loops[self.reduction_depth :])]
        reductions.mark_reduction(self.reduction_vars)
        if len(self.affine_yield) > 0:
            vars = ', '.join([f"%{name}" for name, _ in self.affine_yield.items()])
            reduced_shapes = ', '.join([f"{shape}" for _, shape in self.affine_yield.items()])
            self.stores.writeline(f"affine.yield {vars} : {reduced_shapes}")
        with contextlib.ExitStack() as stack:
            for loop in loops.loops:
                loop_lines = loop.lines()
                if loop_lines is None:
                    return
                code.writelines(loop_lines)
                stack.enter_context(code.indent())
            with contextlib.ExitStack() as stack_outer:
                code.splice(self.reduction_prefix)
                with contextlib.ExitStack() as stack:
                    for reduction in reductions.loops:
                        reduction_lines = reduction.lines()
                        if reduction_lines is None:
                            return
                        code.writelines(reduction_lines)
                        stack.enter_context(code.indent())
                    code.splice(self.loads)
                    code.splice(self.compute)
                    code.splice(self.stores)
                code.splice(self.reductions_suffix)
        code.writeline(f"return")
        return code

    def codegen_kernel(self, kernel_name):
        wrapper = V.graph.wrapper_code
        arg_defs, _, _, _ = self.args.mlir_argdefs()
        code = self._codegen_kernel(arg_defs, kernel_name)
        return code.getvalue()

    def meta_kernel(self):
        wrapper = V.graph.wrapper_code
        _, _, arg_attributes, _ = self.args.mlir_argdefs()
        wrapper.add_import_once('\nprint(f\'Wrapper Codegen Path = {__file__}\')')
        wrapper.add_import_once(f'\nfrom extension_codecache import CustomAsyncCompile')
        wrapper.add_import_once(f'\ncustom_async_compile = CustomAsyncCompile()')
        # Dump loop and load/store information
        wrapper.add_import_once(f"arg_attributes = {arg_attributes}")


    def call_kernel(self, kernel_name):
        wrapper = V.graph.wrapper_code
        _, call_args, _, _ = self.args.mlir_argdefs()
       # generate the code to call this
        wrapper.generate_kernel_call(kernel_name, call_args, cuda=False)

    def _codegen_kernel(self, arg_defs, kernel_name):
        arg_defs = ",\n".ljust(25).join(arg_defs)
        code = common.BracesBuffer()

        code.splice(self.global_vars)
        #TODO:. kernel name custom
        kernel_decl_name = kernel_name if V.graph.cpp_wrapper else "kernel"
        code.writeline(f'func.func @{kernel_decl_name}({arg_defs})')
        with code.indent():
            for old, new in self.args.aliases():
                code.writeline(f"auto {old} = {new};")
            # Loop body part
            code.splice(self.codegen_init())
            code.splice(self.codegen_loops())
        return code

    def adjust_tile_size(self):
        if len(self.itervars) == 1:
            self.tile_desc.n_col = self.tile_desc.get_tile_size()
            self.tile_desc.n_row = 1

    def set_ranges(self, lengths, reduction_lengths):
        if self.call_ranges:
            assert self.call_ranges == tuple(lengths) + tuple(
                reduction_lengths
            ), f"{self.call_ranges} == {tuple(lengths)} + {tuple(reduction_lengths)}"
            assert self.reduction_depth == len(lengths)
        else:
            self.call_ranges = tuple(lengths) + tuple(reduction_lengths)
            self.ranges = [self.rename_indexing(x) for x in self.call_ranges]
            self.itervars = [sympy.Symbol(f"index{n}") for n in range(len(self.ranges))]
            self.reduction_depth = len(lengths)

        # Adjust time size when it is vector
        self.adjust_tile_size()

        # TODO.
        #if len(self.itervars):
        #    if self.tiling_idx < self.reduction_depth and len(reduction_lengths) > 0:#self.reduction_depth > 0:
        #        self.ranges[-1] = (self.ranges[-1] + self.tile_desc.n_row - 1) // self.tile_desc.n_row * self.tile_desc.n_row
        #        if len(self.itervars) > 1:
        #            self.ranges[-2] = (self.ranges[-2] + self.tile_desc.n_col - 1) // self.tile_desc.n_col * self.tile_desc.n_col
        #    else:
        #        self.ranges[-1] = (self.ranges[-1] + self.tile_desc.n_col - 1) // self.tile_desc.n_col * self.tile_desc.n_col
        #        if len(self.itervars) > 1:
        #            self.ranges[-2] = (self.ranges[-2] + self.tile_desc.n_row - 1) // self.tile_desc.n_row * self.tile_desc.n_row
        return (
            self.itervars[: self.reduction_depth],
            self.itervars[self.reduction_depth :],
        )

    def get_scratchpad_buffer(self, dtype, name, tile_row, tile_col, dram_tile_shape, code_buffer):
        c_type = mlir_common.DTYPE_TO_C[dtype]
        mlir_type = mlir_common.DTYPE_TO_MLIR[dtype]
        if dtype == torch.bool:
            mapping = self.map_cse.generate(self.global_vars, f"affine_map<({indices}) -> ({indices} floordiv 8)>")
            indices = self.cse.generate(self.loads, f"affine.apply #{mapping}(%{indices})") # FIXME. Only loads?
        if name not in self.global_vars_set:
            # Add definition to header
            self.header.writeline(f"{c_type} {name}_spad[{tile_row}][{tile_col}] __attribute__ ((section(\".spad\")));")
            self.global_vars_set.add(name)
            self.global_vars.writeline(f"memref.global @{name}_spad : memref<{dram_tile_shape}x{mlir_type}, 1>")
        buffer = self.cse.generate(code_buffer, f"memref.get_global @{name}_spad : memref<{dram_tile_shape}x{mlir_type}, 1>")
        return buffer

@dataclasses.dataclass
class LoopLevel:
    var: sympy.Expr
    size: sympy.Expr
    idx: int
    start: int = 0
    tile_row: int = 4
    tile_col: int = 4
    reduction_vars: Dict[str, str] = None

    def lines(self):
        step = 1
        if self.idx == 0:
            step = self.tile_row
        elif self.idx == 1:
            step = self.tile_col
        if self.reduction_vars:
            acc = ', '.join([f"%{acc.name}" for acc in self.reduction_vars.keys()])
            args = ', '.join([f"%{iter.name} = %{init.name}" for (_, iter, init, _) in self.reduction_vars.values()])
            dtype = ', '.join([f"{dtype}" for (_, _, _, dtype) in self.reduction_vars.values()])
            line = f"{acc} = affine.for %{self.var} = {self.start} to {self.size} step {step} iter_args({args}) -> ({dtype})"
        else:
            line = f"affine.for %{self.var} = {self.start} to {self.size} step {step}"

        return [line]

@dataclasses.dataclass
class LoopNest:
    loops: List[LoopLevel]

    def __bool__(self):
        return bool(self.loops)

    def mark_reduction(self, reduction_vars):
        for loop in self.loops:
            loop.reduction_vars = reduction_vars

    def mark_parallel(self, par_depth):
        loops = self.loops
        loops[0].parallel = par_depth
        for i in range(1, par_depth):
            loops[i].collapsed = True
        loops[0].simd = loops[par_depth - 1].simd

class MLIRScheduling(BaseScheduling):
    count = 0
    target_kernel = MLIRKernel
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self._scheduling = cpp.CppScheduling(scheduler)

    def can_fuse_vertical(self, node1, node2):
        return False

    def can_fuse_horizontal(self, node1, node2):
        return False

    def group_fn(self, sizes):
        return tuple(tuple(map(V.graph.sizevars.simplify, s)) for s in sizes)

    def codegen_nodes(self, nodes):
        _, (group, reduction_group) = max(
            nodes, key=lambda x: int(x.is_reduction())
        ).group
        ex_kernel = self.target_kernel()

        kernel_name = f"extension_kernel_{self.count}"
        self.count += 1
        src_code = ex_kernel.codegen_nodes(nodes, kernel_name)
        self.define_kernel(src_code, kernel_name, ex_kernel.vector_lane, ex_kernel.tile_desc.get_axis_and_tile_info(), ex_kernel.spad_info)
        ex_kernel.call_kernel(kernel_name)
        _, args, _, _ = ex_kernel.args.mlir_argdefs()
        args = ", ".join(args)
        if (bool(os.getenv(BackendSimulator.BACKENDSIM_EAGER_MODE, False))):
            V.graph.wrapper_code.writeline(
                f"yield ({kernel_name}, ({args}))"
            )
    def codegen_sync(self):
        pass

    def flush(self):
        self._scheduling.flush()

    def define_function(self, kernel):
        code = kernel.def_function()
        if code is not None:
            wrapper = V.graph.wrapper_code
            wrapper.header.writeline(code)

    def define_kernel(self, src_code, kernel_name, vector_lane, tile_size, spad_info):
        wrapper = V.graph.wrapper_code
        if src_code in wrapper.src_to_kernel:
            kernel_name = wrapper.src_to_kernel[src_code]
        else:
            wrapper.src_to_kernel[src_code] = kernel_name

            codecache_def = IndentedBuffer()
            codecache_def.writeline(f"custom_async_compile.mlir('''{src_code}''', ")
            codecache_def.writeline(f"vectorlane_size={vector_lane},")
            codecache_def.writeline(f"tile_size={tile_size},")
            codecache_def.writeline(f"spad_info={spad_info},")
            codecache_def.writeline("arg_attributes=arg_attributes)")
            wrapper.define_kernel(kernel_name, codecache_def.getvalue(), cuda=False)
        return kernel_name

    def codegen_template(self, template_node, epilogue_nodes):
        _, (numel, rnumel) = template_node.group

        template_buffer = template_node.node
        kernel, render = template_buffer.make_kernel_render(template_buffer, epilogue_nodes=epilogue_nodes)
        with kernel:
            for node in [template_node, *epilogue_nodes]:
                node.mark_run()
            src_code = render()

        with V.set_kernel_handler(kernel):
            node_schedule = [template_node, *epilogue_nodes]
            kernel.meta_kernel()
            kernel_name = self.define_kernel(src_code, kernel.kernel_name, kernel.vector_lane, (kernel.vector_lane, kernel.vector_lane), kernel.spad_info)
            self.define_function(kernel)
        kernel.call_kernel(kernel_name)
        _, args, _, _ = kernel.args.mlir_argdefs()
        args = ", ".join(args)
        if (bool(os.getenv(BackendSimulator.BACKENDSIM_EAGER_MODE, False))):
            V.graph.wrapper_code.writeline(
                f"yield ({kernel_name}, ({args}))"
            )