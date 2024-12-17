# Owner(s): ["module: inductor"]
import os
import shutil
import sys
import time
import contextlib
import unittest
import copy
import numpy as np
import matplotlib.pyplot as plt
import argparse

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch._dynamo
import torch.utils.cpp_extension
from torch._inductor import config

try:
    from PyTorchSimFrontend.mlir.mlir_codegen_backend import (
        MLIRScheduling,
        ExtensionWrapperCodegen,
    )
except ImportError:
    from .PyTorchSimFrontend.mlir.mlir_codegen_backend import (
        MLIRScheduling,
        ExtensionWrapperCodegen,
    )

from torch._C import FileCheck
from torch._inductor import metrics
from torch._inductor.codegen.common import (
    get_scheduling_for_device,
    get_wrapper_codegen_for_device,
    register_backend_for_device,
)
from torch.testing._internal.common_utils import IS_MACOS
from torch.testing._internal.common_utils import TestCase as TorchTestCase
from test_extension_backend import DecoderBlock, MLP

def remove_build_path():
    if sys.platform == "win32":
        # Not wiping extensions build folder because Windows
        return
    default_build_root = torch.utils.cpp_extension.get_default_build_root()
    if os.path.exists(default_build_root):
        shutil.rmtree(default_build_root, ignore_errors=True)

def apply_random_zero(tensor, zero_prob, block_size=8):
    if not 0 <= zero_prob <= 1:
        raise ValueError("zero_prob must be between 0 and 1.")

    # Generate a random mask with the same shape as the tensor
    mask = torch.rand([tensor.shape[0]//block_size, tensor.shape[1]//block_size]) > zero_prob
    mask = mask.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)
    # Apply the mask to the tensor (set elements to 0 where mask is False)
    return tensor * mask

def count_zeros_in_tensor_list(tensor_list):
    total_zeros = 0
    total_elements = 0
    for tensor in tensor_list:
        zeros_in_tensor = (tensor == 0).sum().item()
        total_zeros += zeros_in_tensor

        total_elements += tensor.numel()
    zero_ratio = total_zeros / total_elements if total_elements > 0 else 0
    print("Sparsity: ", zero_ratio * 100, "%")
    return total_zeros, total_elements, zero_ratio

def test_result(name, out, cpu_out, rtol=1e-4, atol=1e-4):
    message = f"|{name} Test Passed|"
    if torch.allclose(out.cpu(), cpu_out, rtol=rtol, atol=atol):
        print("-" * len(message))
        print(message)
        print("-" * len(message))
    else:
        print("custom out: ", out.cpu())
        print("cpu out: ", cpu_out)

def test_dec_inf(device, sparsity=0.0, block=8):
    torch.manual_seed(0)
    decoder_block = DecoderBlock(768, 12)
    cpu_query = torch.randn(512, 768)
    query = cpu_query.clone().to(device=device)

    cpu_y = decoder_block(cpu_query)
    with torch.no_grad():
        decoder_block.multihead_attn.linears[0].weight.copy_(apply_random_zero(decoder_block.multihead_attn.linears[0].weight, sparsity, block_size=block))
        decoder_block.multihead_attn.linears[1].weight.copy_(apply_random_zero(decoder_block.multihead_attn.linears[1].weight, sparsity, block_size=block))
        decoder_block.multihead_attn.linears[2].weight.copy_(apply_random_zero(decoder_block.multihead_attn.linears[2].weight, sparsity, block_size=block))
        decoder_block.multihead_attn.linears[3].weight.copy_(apply_random_zero(decoder_block.multihead_attn.linears[3].weight, sparsity, block_size=block))
        decoder_block.ffn1.weight.copy_(apply_random_zero(decoder_block.ffn1.weight, sparsity, block_size=block))
        decoder_block.ffn2.weight.copy_(apply_random_zero(decoder_block.ffn2.weight, sparsity, block_size=block))

    count_zeros_in_tensor_list([
        decoder_block.multihead_attn.linears[0].weight,
        decoder_block.multihead_attn.linears[1].weight,
        decoder_block.multihead_attn.linears[2].weight,
        decoder_block.multihead_attn.linears[3].weight,
        decoder_block.ffn1.weight,
        decoder_block.ffn2.weight
    ])

    decoder_block.to(device=device)
    opt_fn = torch.compile()(decoder_block)
    y = opt_fn(query)
    test_result("MLP Forward", y, cpu_y)

def test_mlp_inf(device, batch_size=64, input_size=64, hidden_size=32, output_size=8, sparsity=0.0, block=8):
    torch.manual_seed(0)
    input = torch.randn(batch_size, input_size)
    x1 = copy.deepcopy(input).to(device=device)
    x2 = copy.deepcopy(input).to("cpu")
    target = torch.randn(batch_size, output_size)
    model = MLP(input_size, hidden_size, output_size)
    with torch.no_grad():
        model.linear1.weight.copy_(apply_random_zero(model.linear1.weight, sparsity, block_size=block))
        model.linear2.weight.copy_(apply_random_zero(model.linear2.weight, sparsity, block_size=block))
    count_zeros_in_tensor_list([model.linear1.weight, model.linear2.weight])
    model.requires_grad = False
    model.to(device=device)
    opt_fn = torch.compile()(model)
    y = opt_fn(x1)
    cpu_model = copy.deepcopy(model).to("cpu")
    cpu_model.requires_grad = False
    cpu_y = cpu_model(x2)
    test_result("MLP Forward", y, cpu_y)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count zeros in tensors from command-line arguments.")
    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.8
    )
    parser.add_argument(
        "--block",
        type=int,
        default=8
    )
    args = parser.parse_args()

    from Scheduler.scheduler import ExecutionEngine
    module = ExecutionEngine.setup_device()
    device = module.custom_device()

    #test_dec_inf(device, sparsity=args.sparsity, block=args.block)
    test_mlp_inf(device, batch_size=64, input_size=784, hidden_size=512, output_size=256, sparsity=args.sparsity, block=args.block)
