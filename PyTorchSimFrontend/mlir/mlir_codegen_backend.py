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
    def add(operand1, operand2, tile_size=16):
        shape = f"vector<{tile_size}xf32>" if tile_size > 1 else "f32"
        return f'arith.addf %{operand1}, %{operand2} : {shape}'

    @staticmethod
    def sub(operand1, operand2, tile_size=16):
        shape = f"vector<{tile_size}xf32>" if tile_size > 1 else "f32"
        return f'arith.subf %{operand1}, %{operand2} : {shape}'

    @staticmethod
    def mul(operand1, operand2, tile_size=16):
        shape = f"vector<{tile_size}xf32>" if tile_size > 1 else "f32"
        return f'arith.mulf %{operand1}, %{operand2} : {shape}'

    @staticmethod
    def div(operand1, operand2, tile_size=16):
        shape = f"vector<{tile_size}xf32>" if tile_size > 1 else "f32"
        return f'arith.divf %{operand1}, %{operand2} : {shape}'

    @staticmethod
    def truediv(operand1, operand2, tile_size=16):
        shape = f"vector<{tile_size}xf32>" if tile_size > 1 else "f32"
        return f'arith.divf %{operand1}, %{operand2} : {shape}'

    @staticmethod
    def constant(value, dtype, tile_size=16):
        dtype = mlir_common.DTYPE_TO_MLIR[dtype]
        return f'arith.constant {format(value, ".6f")} : {dtype}'

    @staticmethod
    def exp(operand, tile_size=16):
        shape = f"vector<{tile_size}xf32>" if tile_size > 1 else "f32"
        return f'math.exp %{operand} : {shape}'

    @staticmethod
    def maximum(operand1, operand2, tile_size=16):
        shape = f"vector<{tile_size}xf32>" if tile_size > 1 else "f32"
        return f'arith.maximumf %{operand1}, %{operand2} : {shape}'

    @staticmethod
    def sqrt(x, tile_size=16):
        shape = f"vector<{tile_size}xf32>" if tile_size > 1 else "f32"
        return f'math.sqrt %{x} : {shape}'

    @staticmethod
    def ne(operand1, operand2, tile_size=16):
        shape = f"vector<{tile_size}xi1>" if tile_size > 1 else "i1"
        return f'arith.cmpi one, %{operand1}, %{operand2} : {shape}'

    @staticmethod
    def le(operand1, operand2, tile_size=16):
        shape = f"vector<{tile_size}xf32>" if tile_size > 1 else "f32"
        return f'arith.cmpf ole, %{operand1}, %{operand2} : {shape}'

    @staticmethod
    def relu(x, tile_size=16):
        return ops.maximum(x, ops.constant(0.0, torch.float32))

    @staticmethod
    def sigmoid(x, tile_size=16):
        one = ops.constant(1, torch.float32)
        return ops.truediv(one, ops.add(one, ops.exp(ops.neg(x))))

    @staticmethod
    def neg(x, tile_size=16):
        shape = f"vector<{tile_size}xf32>" if tile_size > 1 else "f32"
        return f'arith.negf %{x} : {shape}'

    @staticmethod
    def where(condition, x, y, tile_size=16):
        shape = f"vector<{tile_size}xf32>" if tile_size > 1 else "f32"
        cond_shape = f"vector<{tile_size}xi1>," if tile_size > 1 else ""
        return f"arith.select %{condition}, %{x}, %{y} : {cond_shape} {shape}"

    @staticmethod
    def logical_not(operand, tile_size=16):
        tile_size=16
        shape = f"vector<{tile_size}xf32>" if tile_size > 1 else "f32"
        result_shape = f"vector<{tile_size}xi1>" if tile_size > 1 else "i1"
        assert(0)
        return f"arith.cmpf oeq, %{operand}, %zero_vec{tile_size} : {shape} -> {result_shape}"
    def rsqrt(x, tile_size=16):
        shape = f"vector<{tile_size}xf32>" if tile_size > 1 else "f32"
        return f'math.rsqrt %{x} : {shape}'

    @staticmethod
    def pow(a, b, tile_size=16):
        shape = f"vector<{tile_size}xf32>" if tile_size > 1 else "f32"
        return f"math.powf %{a}, %{b} : {shape}"

    @staticmethod
    def log(x, tile_size=16):
        shape = f"vector<{tile_size}xf32>" if tile_size > 1 else "f32"
        return f'math.log %{x} : {shape}'

    @staticmethod
    def reciprocal(a, tile_size=16):
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
        self.loop_info = {}
        self.load_desc = {}
        self.store_desc = {}
        self.tiling_indices = [0, 1]
        self.tile_shape = f"{self.tile_row}x{self.tile_col}"
        self.tile_size = self.tile_row * self.tile_col
        self.tile_col_per_lane = self.tile_col // self.vector_lane
        self.tile_size_per_lane = self.tile_size // self.vector_lane
        self.dma_cache = {}
        self.dma_counter = 1
        self.reduction_idx = {}
        self.affine_yield = {}
        self.welford_reduce_out = None
        self.reduce_iterator = {}
        self.is_transpose = False

        # is_transposed
        if len(V.graph.graph_inputs) == 1 and len(V.graph.buffers):
            input_shape = list(V.graph.graph_inputs.values())[0].layout.size
            output_shape = V.graph.buffers[0].layout.size
            if input_shape[-1] == output_shape[-2] and input_shape[-2] == output_shape[-1]:
                self.is_transpose = True

    def get_constant_vector(self, expr):
        constant_vector = [int(expr.coeff(var)) for var in self.itervars]
        return constant_vector

    def get_dma_info(self, cv, name, index, dtype, is_store):
        tile_size_per_lane = min(self.tile_size_per_lane, self.buffer_types[name][1])
        chunk_size = tile_size_per_lane if self.tiling_idx >= self.reduction_depth else self.tile_col_per_lane
        stride = self.tile_col if len(self.itervars) == 1 else reduce(mul, cv, 1)
        is_col_major = 1 if self.tiling_idx >= self.reduction_depth and len(self.itervars) > 1 else 0

        pattern = r"(\d+)x(\d+)"
        matches = re.search(pattern, self.tile_shape)
        t_row = matches.group(1)
        t_col = matches.group(2)
        tile_shape = self.tile_shape
        chunk_size = self.tile_size_per_lane if len(self.itervars) == 1 else chunk_size
        if len(self.itervars) == 1 and index in self.reduction_idx:
            chunk_size = self.tile_size
            is_col_major = 0
            stride = self.tile_size
            t_row = 1
            t_col = self.tile_size
            tile_size_per_lane = min(self.tile_size, self.buffer_types[name][1])
        if len(self.itervars) > 1:
            if cv[1] == 0 or cv[0] == 0:
                chunk_size = self.tile_col_per_lane
                is_col_major = 0
                stride = 0
            if cv[1] == 0:
                t_row = self.vector_lane
                t_col = tile_size_per_lane
                if self.reduction_depth > 1:
                    chunk_size = tile_size_per_lane # TODO: remove this?
                    is_col_major = 1
                    t_row = tile_size_per_lane
                    t_col = self.vector_lane
        elif len(self.itervars) == 0:
            chunk_size = self.tile_size
            is_col_major = 0
            stride = self.tile_size
            tile_size_per_lane = min(self.tile_size, self.buffer_types[name][1])
        tile_shape = f"{t_row}x{t_col}"
        if self.is_transpose and is_store:
            is_col_major = 1 if is_col_major == 0 else 0
            tile_shape = f"{t_col}x{t_row}"
            chunk_size = tile_size_per_lane if chunk_size < stride else self.tile_col_per_lane
        assert(not (dtype==torch.bool and chunk_size < 8))
        chunk = chunk_size << 1 | is_col_major
        return stride, chunk, tile_shape, tile_size_per_lane

    def parse_indices(self, expr):
        if len(expr.args) == 0:
            return expr, expr
        for itervars in self.itervars:
            new_coeff = ((expr.coeff(itervars) + self.tile_col - 1) // self.tile_col) * self.tile_col
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

        def select_tiling_indices():
            all_index = []
            for node in nodes:
                rw = dependencies.extract_read_writes(node._body, *node._sizes)
                all_index += [dep.index for dep in itertools.chain(rw.reads, rw.writes)]
            contig_vars = set()
            contig_vars_list = []
            non_contig_stride_const = set()
            non_contig_stride_other = set()
            for index in all_index:
                for var in index.free_symbols:
                    if not re.search(r"^d\d+$", var.name):
                        continue
                    stride = cpp.stride_at(var, index)
                    if stride == 1:
                        contig_vars.add(int(var.name[1:]))
                        contig_vars_list.append(int(var.name[1:]))
                    elif all(s.name.startswith("s") for s in stride.free_symbols):
                        non_contig_stride_const.add(int(var.name[1:]))
                    else:
                        non_contig_stride_other.add(int(var.name[1:]))
            contig_only = (
                contig_vars - non_contig_stride_const - non_contig_stride_other
            )
            if len(contig_vars) == 0:
                # no contiguous vars
                return [len(self.itervars) - 1]
            if contig_only:
                return sorted(contig_only)[-1:]
            contig_and_const_stride = (
                contig_vars & non_contig_stride_const
            ) - non_contig_stride_other
            contig_vars_sorted = sorted(contig_vars)
            if (
                len(contig_vars_sorted) == 2
                and contig_vars_sorted[-1] in contig_and_const_stride
                and contig_vars_sorted[-1] == len(self.itervars) - 1
            ):
                return contig_vars_sorted
            return sorted(contig_vars_sorted, key=contig_vars_list.count)[-1:]

        self.set_ranges(group, reduction_group)
        self.tiling_indices = select_tiling_indices()
        with self as kernel:
            for node in nodes:
                vars, reduction_vars = kernel.set_ranges(group, reduction_group)
                self.args = mlir_common.MLIRKernelArgs(self.tile_row, self.tile_col)
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
        cv = self.get_constant_vector(index)
        stride, chunk, tile_shape, tile_size_per_lane = self.get_dma_info(cv, name, index, dtype, 0)
        self.header.writeline(f"{mlir_common.DTYPE_TO_C[dtype]} {name}_spad[{self.tile_row}][{self.tile_col}] __attribute__ ((section(\".spad\")));")
        if dtype == torch.bool:
            mapping = self.map_cse.generate(self.global_vars, f"affine_map<({indices}) -> ({indices} floordiv 8)>")
            indices = self.cse.generate(self.loads, f"affine.apply #{mapping}(%{indices})")
        if name not in self.global_vars_set:
            self.global_vars_set.add(name)
            self.global_vars.writeline(f"memref.global @{name}_spad : memref<{tile_shape}x{type_name}, 1>")
        buffer = self.cse.generate(self.loads, f"memref.get_global @{name}_spad : memref<{tile_shape}x{type_name}, 1>")

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
        code = f"affine.dma_start %{var}[{prefix}{indices}], %{buffer}[0, 0], %{name}_tag[0], %c{dmaType}, %c{stride}, %c{chunk} : memref<{self.buffer_types[name][1]}x{type_name}>, memref<{tile_shape}x{type_name}, 1>, memref<1xi32>"
        self.cse.generate(self.loads, code, assignment = False) # FIXME: assignment = False does not support caching

        operation = "affine.vector_load" if tile_size_per_lane > 1 else "affine.load"
        shape = f", vector<{tile_size_per_lane}x{type_name}>" if tile_size_per_lane > 1 else ""
        line = f"{operation} %{buffer}[0, 0] : memref<{tile_shape}x{type_name}, 1>{shape}"
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
        cv = self.get_constant_vector(index)
        stride, chunk, tile_shape, tile_size_per_lane = self.get_dma_info(cv, name, index, dtype, 1)
        self.header.writeline(f"{mlir_common.DTYPE_TO_C[dtype]} {name}_spad[{self.tile_row}][{self.tile_col}] __attribute__ ((section(\".spad\")));")
        if dtype == torch.bool:
            mapping = self.map_cse.generate(self.global_vars, f"affine_map<({indices}) -> ({indices} floordiv 8)>")
            indices = self.cse.generate(self.loads, f"affine.apply #{mapping}(%{indices})")
        if name not in self.global_vars_set:
            self.global_vars_set.add(name)
            self.global_vars.writeline(f"memref.global @{name}_spad : memref<{tile_shape}x{type_name}, 1>")
        buffer = self.cse.generate(self.stores, f"memref.get_global @{name}_spad : memref<{tile_shape}x{type_name}, 1>")

        # MVOUT Encoding
        dmaType = 3 # MVIN 2, MVIN2 1, MVIN3 14, MVOUT 3
        self.consts.add(dmaType)
        self.consts.add(stride)
        self.consts.add(chunk)

        operation = "affine.vector_store" if tile_size_per_lane > 1 else "affine.store"
        shape = f", vector<{tile_size_per_lane}x{type_name}>" if tile_size_per_lane > 1 else ""
        line = f"{operation} %{value}, %{buffer}[0, 0] : memref<{tile_shape}x{type_name}, 1>{shape}"
        self.cse.generate(self.stores, line, assignment = False)
        self.tags.add(f"{name}_tag")
        code = f"affine.dma_start %{buffer}[0, 0], %{var}[{prefix}{indices}], %{name}_tag[0], %c{dmaType}, %c{stride}, %c{chunk} : memref<{tile_shape}x{type_name}, 1>, memref<{self.buffer_types[name][1]}x{type_name}>, memref<1xi32>"
        self.cse.generate(self.stores, code, assignment = False)

    def reduction(self, dtype, src_dtype, reduction_type, value):
        argmax_or_argmin = reduction_type in {"argmax", "argmin"}
        if argmax_or_argmin:
            raise NotImplementedError() #TODO: argmin, argmax
        elif is_welford_reduction(reduction_type):
            if reduction_type == "welford_combine":
                pass
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
            shape = f"vector<{self.tile_size}x{type_name}>"
            reduced_shape = type_name
            self.reduction_prefix.writeline(f"%{init} = arith.constant {reduction_init(reduction_type, dtype)} : {type_name}")
            if len(self.itervars) > 1:
                self.reduction_prefix.writeline(f"%{init_vec} = vector.broadcast %{init} : {type_name} to vector<{self.tile_col_per_lane}x{type_name}>")
                vec_len = self.tile_col if self.tiling_idx >= self.reduction_depth else self.tile_row
                value = self.cse.generate(self.compute, f"vector.shape_cast %{value} : vector<{self.tile_size_per_lane}x{type_name}> to vector<{vec_len}x{self.tile_col_per_lane}x{type_name}>")
                shape = f"vector<{vec_len}x{self.tile_col_per_lane}x{type_name}>"
                acc_var = init_vec
                reduced_shape = f"vector<{self.tile_col_per_lane}x{type_name}>"
            self.reduction_vars[acc] = (reduction_type, iterator, acc_var, reduced_shape)
            out = self.cse.generate(self.compute, reduction_combine_vec(reduction_type, value, iterator, axis=0, shape=shape, reduced_shape=reduced_shape))
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
        tile_row = self.vector_lane if len(self.itervars) > 1 else 1
        tile_col = self.tile_col_per_lane if len(self.itervars) > 1 else 1
        self.header.writeline(f"{mlir_common.DTYPE_TO_C[dtype]} {name}_spad[{tile_row}][{tile_col}] __attribute__ ((section(\".spad\")));")
        if name not in self.global_vars_set:
            self.global_vars_set.add(name)
            self.global_vars.writeline(f"memref.global @{name}_spad : memref<{tile_row}x{tile_col}x{type_name}, 1>")
        buffer = self.cse.generate(self.reductions_suffix, f"memref.get_global @{name}_spad : memref<{tile_row}x{tile_col}x{type_name}, 1>")
        if self.welford_reduce_out is not None:
            sum, sqr_sum, _ = self.welford_reduce_out

            # mean
            self.cse.generate(self.reductions_suffix, f"%f{self.buffer_types[name][1]} = arith.constant {float(self.buffer_types[name][1])} : f32", assignment=False)
            divider_vec = self.cse.generate(self.reductions_suffix, f"vector.broadcast %f{self.buffer_types[name][1]} : f32 to vector<{self.tile_col_per_lane}x{type_name}>")
            mean = self.cse.generate(self.reductions_suffix, f"arith.divf %{sum}, %{divider_vec} : vector<{self.tile_col_per_lane}x{type_name}>")

            # m2 = (E(X^2) - E(X)^2) * N
            sqr_mean = self.cse.generate(self.reductions_suffix, f"arith.divf %{sqr_sum}, %{divider_vec} : vector<{self.tile_col_per_lane}x{type_name}>")
            mean_sqr = self.cse.generate(self.reductions_suffix, f"arith.mulf %{mean}, %{mean} : vector<{self.tile_col_per_lane}x{type_name}>")
            variance = self.cse.generate(self.reductions_suffix, f"arith.subf %{sqr_mean}, %{mean_sqr} : vector<{self.tile_col_per_lane}x{type_name}>")
            m2 = self.cse.generate(self.reductions_suffix, f"arith.mulf %{variance}, %{divider_vec} : vector<{self.tile_col_per_lane}x{type_name}>")
            if name == "buf0":
                value = mean
            else:
                value = m2

        operation = "affine.vector_store" if self.buffer_types[name][1] > 1 else "affine.store"
        shape = f", vector<{tile_col}x{type_name}>" if self.buffer_types[name][1] > 1 else ""
        line = f"{operation} %{value}, %{buffer}[0, 0] : memref<{tile_row}x{tile_col}x{type_name}, 1>{shape}"
        self.cse.generate(self.reductions_suffix, line, assignment = False)

        # MVOUT Encoding
        dmaType = 3 # MVIN 2, MVIN2 1, MVIN3 14, MVOUT 3
        stride = tile_col
        is_col_major = 0
        chunk_size = tile_col
        chunk = chunk_size << 1 | is_col_major
        self.consts.add(dmaType)
        self.consts.add(stride)
        self.consts.add(chunk)
        self.tags.add(f"{name}_tag")
        code = f"affine.dma_start %{buffer}[0, 0], %{var}[{prefix}{indices}], %{name}_tag[0], %c{dmaType}, %c{stride}, %c{chunk} : memref<{tile_row}x{tile_col}x{type_name}, 1>, memref<{self.buffer_types[name][1]}x{type_name}>, memref<1xi32>"
        self.cse.generate(self.reductions_suffix, code, assignment = False)

    def codegen_init(self):
        code = IndentedBuffer()
        for tag in self.tags:
            code.writeline(f"%{tag} = memref.alloc() : memref<1xi32>")
        for const in self.consts:
            code.writeline(f"%c{const} = arith.constant {const} : index")
        return code

    def codegen_loops(self):
        code = mlir_common.ParallelLoopBuffer()
        # Loop body part
        tile_row, tile_col = self.tile_row, self.tile_col
        if (self.tiling_idx < self.reduction_depth and len(self.reduction_idx) > 0):
            tile_row, tile_col = self.tile_col, self.tile_row
        tile_row = self.tile_size if len(self.itervars) == 1 else tile_row
        loops = [LoopLevel(var, size, idx, tile_row=tile_row, tile_col=tile_col, is_transpose=self.is_transpose) for idx, (var, size) in enumerate(zip(self.itervars, self.ranges))]
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
        wrapper.add_import_once(f"loop_info = {self.loop_info}")
        wrapper.add_import_once(f"load_tile_info = {self.load_desc}")
        wrapper.add_import_once(f"store_tile_info = {self.store_desc}")
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
        # do vertical reduction as the tail loop
        if len(self.itervars) > 1:
            if len(self.tiling_indices) == 1:
                self.tiling_idx = self.tiling_indices[0]
                self.outer_idx = None
            else:
                self.outer_idx, self.tiling_idx = (
                    self.tiling_indices
                    if self.tiling_indices[1] < self.reduction_depth
                    else reversed(self.tiling_indices)
                )
        else:
            self.tiling_idx = self.tiling_indices[0]
            self.outer_idx = None
            self.tile_row = 1
            self.tile_col = self.tile_size
            self.tile_shape = f"{self.tile_row}x{self.tile_col}"
        if self.tiling_idx >= self.reduction_depth and len(reduction_lengths) > 0: #self.reduction_depth > 0:
            self.tile_row, self.tile_col = self.tile_col, self.tile_row
            self.tile_shape = f"{self.tile_row}x{self.tile_col}"
        if len(self.itervars):
            if self.tiling_idx < self.reduction_depth and len(reduction_lengths) > 0:#self.reduction_depth > 0:
                self.ranges[-1] = (self.ranges[-1] + self.tile_row - 1) // self.tile_row * self.tile_row
                if len(self.itervars) > 1:
                    self.ranges[-2] = (self.ranges[-2] + self.tile_col - 1) // self.tile_col * self.tile_col
            else:
                self.ranges[-1] = (self.ranges[-1] + self.tile_col - 1) // self.tile_col * self.tile_col
                if len(self.itervars) > 1:
                    self.ranges[-2] = (self.ranges[-2] + self.tile_row - 1) // self.tile_row * self.tile_row
        return (
            self.itervars[: self.reduction_depth],
            self.itervars[self.reduction_depth :],
        )

@dataclasses.dataclass
class LoopLevel:
    var: sympy.Expr
    size: sympy.Expr
    idx: int
    start: int = 0
    tile_row: int = 4
    tile_col: int = 4
    reduction_vars: Dict[str, str] = None
    is_transpose: bool = False

    def lines(self):
        step = 1
        if self.idx == 0:
            step = self.tile_col if self.is_transpose else self.tile_row
        elif self.idx == 1:
            step = self.tile_row if self.is_transpose else self.tile_col
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
        self.define_kernel(src_code, kernel_name, ex_kernel.vector_lane, (ex_kernel.tile_row, ex_kernel.tile_col), ex_kernel.spad_info)
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
            codecache_def.writeline("loop_info=loop_info,")
            codecache_def.writeline("load_tile_info=load_tile_info,")
            codecache_def.writeline("store_tile_info=store_tile_info,")
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