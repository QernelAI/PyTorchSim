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

import torch
import torch.nn as nn
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

def remove_build_path():
    if sys.platform == "win32":
        # Not wiping extensions build folder because Windows
        return
    default_build_root = torch.utils.cpp_extension.get_default_build_root()
    if os.path.exists(default_build_root):
        shutil.rmtree(default_build_root, ignore_errors=True)

class TestCase(TorchTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._stack = contextlib.ExitStack()
        cls._stack.enter_context(
            config.patch(
                {
                    "debug": True,
                    "debug_index_asserts": True,
                    "cpp.min_chunk_size": 1,
                    "triton.autotune_pointwise": False,  # too slow
                    "implicit_fallbacks": False,
                    "generate_intermediate_hooks": True,
                }
            )
        )

    @classmethod
    def tearDownClass(cls):
        cls._stack.close()
        super().tearDownClass()

    def setUp(self):
        torch._dynamo.reset()
        torch._inductor.metrics.reset()
        super().setUp()
        self._start = time.perf_counter()

    def tearDown(self):
        super().tearDown()
        torch._dynamo.reset()
        if os.environ.get("ERROR_ON_SLOW") == "1":
            elapsed = time.perf_counter() - self._start
            assert elapsed < 120


class ExtensionBackendTests(TestCase):
    module = None

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        # Build Extension
        remove_build_path()
        source_file_path = os.path.dirname(os.path.abspath(__file__))
        source_file = os.path.join(
            source_file_path, "PyTorchSimFrontend/extension_device.cpp"
        )
        cls.module = torch.utils.cpp_extension.load(
            name="extension_device",
            sources=[
                str(source_file),
            ],
            extra_cflags=["-g"],
            verbose=True,
        )

    @classmethod
    def tearDownClass(cls):
        cls._stack.close()
        super().tearDownClass()

        remove_build_path()

    def setUp(self):
        torch._dynamo.reset()
        super().setUp()

        # cpp extensions use relative paths. Those paths are relative to
        # this file, so we'll change the working directory temporarily
        self.old_working_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        assert self.module is not None

    def tearDown(self):
        super().tearDown()
        torch._dynamo.reset()

        # return the working directory (see setUp)
        os.chdir(self.old_working_dir)

    def test_open_device_registration(self):
        torch.utils.rename_privateuse1_backend("extension_device")

        register_backend_for_device(
            "extension_device", MLIRScheduling, ExtensionWrapperCodegen
        )
        self.assertTrue(
            get_scheduling_for_device("extension_device") == MLIRScheduling
        )
        self.assertTrue(
            get_wrapper_codegen_for_device("extension_device")
            == ExtensionWrapperCodegen
        )

        device = self.module.custom_device()
        metrics.reset()

        # torch.set_printoptions(threshold=float('inf'), linewidth=600)

        # element-wise operation TEST
        # test_vectoradd(device)
        # test_reduce_sum(device)

        # conv TEST
        # test_conv2d(device)

        # Backward Uni-Test (Single Perceptron)
        # test_single_perceptron(device)

        # Matmul TEST
        # test_matmul(device)

        # MLP TEST
        # test_mlp(device)

        # Softmax TEST
        # test_softmax(device)

        # ReLU TEST
        # test_ReLU(device)

        # CNN TEST
        # test_CNN(device)

        # LayerNorm TEST
        # test_LayerNorm(device)

        # Multihead Attention Test
        # test_Attention(device)
        # test_MultiAttention(device)

        # BMM Test
        # test_BMM(device)

        # Transpose Test
        # test_Transpose(device)

        # Optimizer Test
        # test_optimizer(device)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(64, 64)
        self.linear2 = nn.Linear(64, 64)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear1(self.relu(x))
        x = self.linear2(self.relu(x))
        x = self.softmax(x)
        return x

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        return x

class LayerNorm(nn.Module):
    def __init__(self, n):
        super(LayerNorm, self).__init__()
        self.ln = nn.LayerNorm(n)
    
    def forward(self, x):
        return self.ln(x)

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiheadAttention, self).__init__()
        multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
    
    def forward(self, x):
        return multihead_attn(x, x, x)

import math
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
class my_MultiheadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(my_MultiheadAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
    
    def attention(self, query, key, value):
        d_k = query.size(-1)
        print("Attention in CPU >")
        print("Score CPU > ")
        print(torch.matmul(query, key.transpose(-2, -1)))

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        p_attn = scores.softmax(dim=-1)
        print("Softmax CPU > ")
        print(p_attn)
        return torch.matmul(p_attn, value), p_attn

    def forward(self, query, key, value):
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(-1, self.h, self.d_k).transpose(0, 1).contiguous()
            for lin, x in zip(self.linears, (query, key, value))
        ]

        if query.device == torch.device("cpu"):
            print("QKV After Linear Projection in CPU >")
            print("CPU Query > ", query)
            print("CPU Query Weight > ", self.linears[0].weight)
            print("CPU Query Bias > ", self.linears[0].bias)
            print("CPU Key > ", key)
            print("CPU Key Weight > ", self.linears[1].weight)
            print("CPU Key Bias > ", self.linears[1].bias)
            print("CPU Value > ", value)
            print("CPU Value Weight > ", self.linears[2].weight)
            print("CPU Value Bias > ", self.linears[2].bias)

        # 2) Apply attention on all the projected vectors in batch.
        # x, self.attn = self.attention(query, key, value)
        # d_k = query.size(-1)

        if query.device == torch.device("cpu"):
            print("Attention in CPU >")
            print("Score CPU > ")
            print(torch.matmul(query, key.transpose(-2, -1)))

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        p_attn = scores.softmax(dim=-1)
        if query.device == torch.device("cpu"):
            print("Softmax CPU > ")
            print(p_attn)
        x = torch.matmul(p_attn, value)

        if query.device == torch.device("cpu"):
            print("Attention Result in CPU >")
            print("CPU X > ", x)

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(0, 1)
            .contiguous()
            .view(-1, self.h * self.d_k)
        )

        if query.device == torch.device("cpu"):
            print("X After Concat in CPU >")
            print("CPU X > ", x)

        del query
        del key
        del value
        return self.linears[-1](x)
class my_Decoder(nn.Module):
    def __init__(self):
        # custom transformer decoder
        super(my_Decoder, self).__init__()
        

def test_vectoradd(device):
    def vectoradd(a, b):
        return a + b
    x = torch.randn(128, 128).to(device=device)
    y = torch.randn(128, 128).to(device=device)
    opt_fn = torch.compile()(vectoradd)
    res = opt_fn(x, y)
    out = vectoradd(x.cpu(), y.cpu())
    if torch.allclose(res.cpu(), out, rtol=1e-4, atol=1e-4):
        print("-----------------------")
        print("|VectorAdd Test Passed|")
        print("-----------------------")
    else:
        print("custom out: ", res.cpu())
        print("cpu out: ", out)

def test_reduce_sum(device):
    def reduce_sum(a, b):
        return torch.sum(a + b, axis=0)
    x = torch.randn(128, 128).to(device=device)
    y = torch.randn(128, 128).to(device=device)
    opt_fn = torch.compile()(reduce_sum)
    res = opt_fn(x, y)
    out = reduce_sum(x.cpu(), y.cpu())
    if torch.allclose(res.cpu(), out, rtol=1e-4, atol=1e-4):
        print("-----------------------")
        print("|ReduceSum Test Passed|")
        print("-----------------------")
    else:
        print("custom out: ", res.cpu())
        print("cpu out: ", out)

def test_single_perceptron(device):
    def perceptron(a, b, c):
        res = a * b + c
        return res

    def weight_update(a, b, lr):
        return a - b * lr
    from sklearn.datasets import make_regression
    X, Y = make_regression(n_samples=128, n_features=1, noise=30, random_state=1)
    input = torch.tensor(X.squeeze(-1), dtype=torch.float32)
    weight = torch.randn(1)
    x1 = copy.deepcopy(input).to(device=device)
    x2 = copy.deepcopy(input).to("cpu")
    w1 = copy.deepcopy(weight).to(device=device)
    w2 = copy.deepcopy(weight).to("cpu")
    target_y = torch.tensor(Y, dtype=torch.float32)
    y1 = copy.deepcopy(target_y).to(device=device)
    y2 = copy.deepcopy(target_y).to("cpu")
    b = torch.randn(1)
    b1 = copy.deepcopy(b).to(device=device)
    b2 = copy.deepcopy(b).to("cpu")
    w1.requires_grad = True
    w2.requires_grad = True
    b1.requires_grad = True
    b2.requires_grad = True
    opt_mlp = torch.compile()(perceptron)
    opt_w = torch.compile()(weight_update)
    opt_loss = torch.compile()(torch.nn.MSELoss())
    lr = torch.tensor(5e-2).to(device=device) # learning rate
    y = opt_mlp(w1, x1, b1)
    loss = opt_loss(y, y1)
    loss.backward()
    cpu_y = perceptron(x2, w2, b2)
    cpu_loss = torch.nn.MSELoss()(cpu_y, y2)
    cpu_loss.backward()
    if torch.allclose(y.cpu(), cpu_y, rtol=1e-4, atol=1e-4):
        print("-------------------------------")
        print("|Single Perceptron Test Passed|")
        print("-------------------------------")
    else:
        print("custom out: ", y.cpu())
        print("cpu out: ", cpu_y)
    if torch.allclose(loss.cpu(), cpu_loss, rtol=1e-4, atol=1e-4):
        print("----------------------------")
        print("|Loss Function Test Passed |")
        print("----------------------------")
    else:
        print("custom loss: ", loss.cpu())
        print("cpu loss: ", cpu_loss)
    if torch.allclose(w1.grad.cpu(), w2.grad, rtol=1e-4, atol=1e-4):
        print("----------------------------")
        print("|Weight Update Test Passed |")
        print("----------------------------")
    else:
        print("custom grad: ", w1.grad.cpu())
        print("cpu grad: ", w2.grad)
    if torch.allclose(b1.grad.cpu(), b2.grad, rtol=1e-4, atol=1e-4):
        print("----------------------------")
        print("|  Bias Update Test Passed |")
        print("----------------------------")
    else:
        print("custom grad: ", b1.grad.cpu())
        print("cpu grad: ", b2.grad)
    # for i in range(50):
    #     y = opt_mlp(w1, x1, b1)
    #     loss = opt_loss(y, y1)
    #     # print(loss.cpu().item()) # check loss
    #     loss.to(device=device)
    #     loss.backward()
    #     with torch.no_grad():
    #         w1.copy_(opt_w(w1, w1.grad, lr))
    #         b1.copy_(opt_w(b1, b1.grad, lr))
    #     w1.grad.zero_()
    #     b1.grad.zero_()
    # # plot input and output on 2D plane, and plot the y = w*x + b line
    # plt.scatter(x1.cpu().numpy(), y1.cpu().numpy(), c='#c80151')
    # x = np.linspace(-3, 3, 100)
    # y = w1.cpu().item() * x + b1.cpu().item()
    # plt.plot(x, y, '-k')
    # plt.show()
    # plt.savefig('result.png')

def test_matmul(device):
    def custom_matmul(a, b):
        # return torch.addmm(bias, a, b)
        return torch.matmul(a, b)
    torch.manual_seed(0)
    input = torch.randn(128, 64)
    weight = torch.randn(64, 32)
    bias = torch.randn(64)
    x1 = input.to(device=device)
    w1 = weight.to(device=device)
    b1 = bias.to(device=device)
    x2 = input.to("cpu")
    w2 = weight.to("cpu")
    b2 = bias.to("cpu")
    opt_fn = torch.compile()(custom_matmul)
    res = opt_fn(x1, w1)
    y = custom_matmul(x2, w2)
    if torch.allclose(res.cpu(), y, rtol=1e-4, atol=1e-4):
        print("----------------------------")
        print("|Matmul Forward Test Passed|")
        print("----------------------------")
    else:
        print("custom out: ", res.cpu())
        print("cpu out: ", y)

def test_mlp(device):
    torch.manual_seed(0)
    input = torch.randn(64, 64)
    x1 = copy.deepcopy(input).to(device=device)
    x2 = copy.deepcopy(input).to("cpu")
    target = torch.randn(64, 64)
    y1 = copy.deepcopy(target).to(device=device)
    y2 = copy.deepcopy(target).to("cpu")
    model = MLP()
    model.requires_grad = True
    model.to(device=device)
    opt_fn = torch.compile()(model)
    y = opt_fn(x1)
    cpu_model = copy.deepcopy(model).to("cpu")
    cpu_model.requires_grad = True
    cpu_y = cpu_model(x2)
    loss_fn = torch.nn.CrossEntropyLoss()
    opt_loss = torch.compile()(loss_fn)
    loss = loss_fn(y, y1)
    cpu_loss = loss_fn(cpu_y, y2)
    loss.backward()
    cpu_loss.backward()
    if torch.allclose(y.cpu(), cpu_y, rtol=1e-4, atol=1e-4):
        print("----------------------------")
        print("| MLP Forward Test Passed  |")
        print("----------------------------")
    else:
        print("custom out: ", y.cpu())
        print("cpu out: ", cpu_y)
    if torch.allclose(loss.cpu(), cpu_loss, rtol=1e-4, atol=1e-4):
        print("----------------------------")
        print("|Loss Function Test Passed |")
        print("----------------------------")
    else:
        print("custom loss: ", loss.cpu())
        print("cpu loss: ", cpu_loss)
    if torch.allclose(model.linear1.weight.grad.cpu(), cpu_model.linear1.weight.grad, rtol=1e-4, atol=1e-4):
        print("----------------------------")
        print("|MLP Backward Test 1 Passed|")
        print("----------------------------")
    else:
        print("custom grad: ", model.linear1.weight.grad.cpu())
        print("cpu grad: ", cpu_model.linear1.weight.grad)
    if torch.allclose(model.linear1.bias.grad.cpu(), cpu_model.linear1.bias.grad, rtol=1e-4, atol=1e-4):
        print("----------------------------")
        print("|MLP Backward Test 2 Passed|")
        print("----------------------------")
    else:
        print("custom grad: ", model.linear1.bias.grad.cpu())
        print("cpu grad: ", cpu_model.linear1.bias.grad)
    if torch.allclose(model.linear2.weight.grad.cpu(), cpu_model.linear2.weight.grad, rtol=1e-4, atol=1e-4):
        print("----------------------------")
        print("|MLP Backward Test 3 Passed|")
        print("----------------------------")
    else:
        print("custom grad: ", model.linear2.weight.grad.cpu())
        print("cpu grad: ", cpu_model.linear2.weight.grad)
    if torch.allclose(model.linear2.bias.grad.cpu(), cpu_model.linear2.bias.grad, rtol=1e-4, atol=1e-4):
        print("----------------------------")
        print("|MLP Backward Test 4 Passed|")
        print("----------------------------")
    else:
        print("custom grad: ", model.linear2.bias.grad.cpu())
        print("cpu grad: ", cpu_model.linear2.bias.grad)

def test_CNN(device):
    torch.manual_seed(0)
    input = torch.randn(1, 8, 64, 64)
    x1 = input.to(device=device)
    x2 = input.to("cpu")
    model = CNN()
    model.to(device=device)
    opt_fn = torch.compile()(model)
    y = opt_fn(x1)
    cpu_model = model.to("cpu")
    cpu_y = cpu_model(x2)
    if torch.allclose(y.cpu(), cpu_y, rtol=2e-1, atol=2e-1):
        print("-------------------------")
        print("|CNN Forward Test Passed|")
        print("-------------------------")
    else:
        print("custom out: ", y.cpu())
        print("cpu out: ", cpu_y)
    print("Max diff > ", torch.max(torch.abs(y.cpu() - cpu_y)))

def test_conv2d(device):
    def custom_conv2d(a, b):
        i_c = a.shape[1]
        o_c = b.shape[0]
        conv2d = nn.Conv2d(i_c, o_c, b.shape[-1], stride=1, padding=0, dilation=1)
        conv2d.weight = nn.Parameter(b)
        return conv2d(a)
    torch.manual_seed(0)
    conv_input = torch.randn(1, 8, 64, 64).to(device=device)
    conv_kernel = torch.randn(16, 8, 3, 3).to(device=device)
    opt_fn = torch.compile()(custom_conv2d)
    res = opt_fn(conv_input, conv_kernel)
    out = custom_conv2d(conv_input.cpu(), conv_kernel.cpu())
    if torch.allclose(res.cpu(), out, rtol=1e-1, atol=1e-1):
        print("----------------------------")
        print("|Conv2d Forward Test Passed|")
        print("----------------------------")
    else:
        print("custom out: ", res.cpu())
        print("cpu out: ", out)
    print("Max diff > ", torch.max(torch.abs(res.cpu() - out)))

def test_softmax(device):
    torch.manual_seed(0)
    input = torch.randn(128, 128)
    x1 = input.to(device=device)
    x2 = input.to("cpu")
    opt_fn = torch.compile()(torch.nn.functional.softmax)
    y = opt_fn(x1, dim=1)
    cpu_y = torch.nn.functional.softmax(x2, dim=1)
    if torch.allclose(y.cpu(), cpu_y, rtol=1e-4, atol=1e-4):
        print("-----------------------------")
        print("|Softmax Forward Test Passed|")
        print("-----------------------------")
    else:
        print("custom out: ", y.cpu())
        print("cpu out: ", cpu_y)

def test_ReLU(device):
    torch.manual_seed(0)
    input = torch.randn(128, 128)
    x1 = input.to(device=device)
    x2 = input.to("cpu")
    opt_fn = torch.compile()(torch.nn.functional.relu)
    y = opt_fn(x1)
    cpu_y = torch.nn.functional.relu(x2)
    if torch.allclose(y.cpu(), cpu_y, rtol=1e-4, atol=1e-4):
        print("--------------------------")
        print("|ReLU Forward Test Passed|")
        print("--------------------------")
    else:
        print("custom out: ", y.cpu())
        print("cpu out: ", cpu_y)

def test_LayerNorm(device):
    torch.manual_seed(0)
    input = torch.randn(64, 64)
    x1 = input.to(device=device)
    x2 = input.to("cpu")
    model = LayerNorm(64)
    model.to(device=device)
    opt_fn = torch.compile()(model)
    y = opt_fn(x1)
    cpu_model = model.to("cpu")
    cpu_y = cpu_model(x2)
    if torch.allclose(y.cpu(), cpu_y, rtol=1e-4, atol=1e-4):
        print("-----------------------------")
        print("|LayerNorm Forward Test Passed|")
        print("-----------------------------")
    else:
        print("custom out: ", y.cpu())
        print("cpu out: ", cpu_y)

def test_Attention(device):
    def attention(query, key, value):
        import math
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        p_attn = scores.softmax(dim=-1)
        return torch.matmul(p_attn, value), p_attn
    
    torch.manual_seed(0)
    query = torch.randn(16, 128).to(device=device)
    key = torch.randn(16, 128).to(device=device)
    value = torch.randn(16, 128).to(device=device)

    opt_fn = torch.compile()(attention)
    res, p_attn = opt_fn(query, key, value)

    cpu_res, cpu_p_attn = attention(query.cpu(), key.cpu(), value.cpu())

    if torch.allclose(res.cpu(), cpu_res, rtol=1e-4, atol=1e-4):
        print("----------------------------")
        print("|Attention Forward Test Passed|")
        print("----------------------------")
    else:
        print("custom out: ", res.cpu())
        print("cpu out: ", cpu_res)

def test_MultiAttention(device):
    torch.manual_seed(0)
    query = torch.randn(16, 128).to(device=device)
    key = torch.randn(16, 128).to(device=device)
    value = torch.randn(16, 128).to(device=device)

    print("Model Input Print >>>>>> ")
    print("Query > ", query.cpu())
    print("Key > ", key.cpu())
    print("Value > ", value.cpu())

    multihead_attn = my_MultiheadAttention(8, 128)
    multihead_attn.to(device=device)
    opt_fn = torch.compile()(multihead_attn)
    res = opt_fn(query, key, value)

    cpu_multihead_attn = multihead_attn.to("cpu")
    cpu_res = cpu_multihead_attn(query.cpu(), key.cpu(), value.cpu())

    if torch.allclose(res.cpu(), cpu_res, rtol=1e-4, atol=1e-4):
        print("----------------------------")
        print("|Multihead Attention Forward Test Passed|")
        print("----------------------------")
    else:
        print("custom out: ", res.cpu())
        print("cpu out: ", cpu_res)

def test_BMM(device):
    def bmm(a, b):
        return torch.bmm(a, b.transpose(1, 2))
    torch.manual_seed(0)
    a = torch.randn(1, 32, 64).to(device=device)
    b = torch.randn(1, 16, 64).to(device=device)
    opt_fn = torch.compile()(bmm)
    res = opt_fn(a, b)
    out = bmm(a.cpu(), b.cpu())
    if torch.allclose(res.cpu(), out, rtol=1e-4, atol=1e-4):
        print("----------------------------")
        print("|BMM Forward Test Passed|")
        print("----------------------------")
    else:
        print("custom out: ", res.cpu())
        print("cpu out: ", out)

def test_Transpose(device):
    def transpose(a):
        return a.transpose(0, 1).contiguous()
    torch.manual_seed(0)
    # x = torch.randn(16, 32).to(device=device)
    x = torch.arange(16*32).reshape(16, 32).float().to(device=device)
    opt_fn = torch.compile()(transpose)
    res = opt_fn(x)
    out = transpose(x.cpu())
    if torch.allclose(res.cpu(), out, rtol=1e-4, atol=1e-4):
        print("----------------------------")
        print("|Transpose Forward Test Passed|")
        print("----------------------------")
    else:
        print("custom out: ", res.cpu())
        print("cpu out: ", out)

def test_optimizer(device):
    torch.manual_seed(0)
    model = MLP().to(device=device)
    cpu_model = copy.deepcopy(model).to("cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    cpu_optimizer = torch.optim.Adam(cpu_model.parameters(), lr=0.001)
    opt_step = torch.compile()(optimizer.step)
    input = torch.randn(16, 16)
    x1 = copy.deepcopy(input).to(device=device)
    x2 = copy.deepcopy(input).to("cpu")
    y = model(x1)
    cpu_y = cpu_model(x2)
    loss = y.sum()
    cpu_loss = cpu_y.sum()
    optimizer.zero_grad()
    cpu_optimizer.zero_grad()
    loss.backward()
    cpu_loss.backward()
    opt_step()
    cpu_optimizer.step()
    if torch.allclose(model.linear1.weight.cpu(), cpu_model.linear1.weight, rtol=1e-4, atol=1e-4):
        print("-----------------------")
        print("|Optimizer Test Passed|")
        print("-----------------------")
    else:
        print("custom out: ", model.linear1.weight.cpu())
        print("cpu out: ", cpu_model.linear1.weight)

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests
    from torch.testing._internal.inductor_utils import HAS_CPU

    if HAS_CPU and not IS_MACOS:
        run_tests(needs="filelock")