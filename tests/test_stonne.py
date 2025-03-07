import torch
import torch._dynamo
import torch.utils.cpp_extension
import random
import numpy as np

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def apply_pruning(tensor, sparsity):
    mask = torch.rand_like(tensor) >= sparsity
    tensor *= mask

def test_result(name, out, cpu_out, rtol=1e-4, atol=1e-4):
    message = f"|{name} Test Passed|"
    if torch.allclose(out.cpu(), cpu_out, rtol=rtol, atol=atol):
        print("-" * len(message))
        print(message)
        print("-" * len(message))
    else:
        print("custom out: ", out.cpu())
        print("cpu out: ", cpu_out)
        exit(1)

def sparse_matmul(a, b):
    return torch.sparse.mm(a, b)

def test_sparse_mm(device, input_size=128, hidden_size=128, output_size=128):
    torch.manual_seed(0)
    input = torch.randn(input_size, hidden_size)
    weight = torch.randn(hidden_size, output_size)
    x1 = input.to(device=device)
    w1 = weight.to(device=device)
    opt_fn = torch.compile(dynamic=False)(sparse_matmul)
    res = opt_fn(x1, w1)
    cpu_res = sparse_matmul(input.cpu(), weight.cpu())
    test_result("spmm", res, cpu_res)
 
 
if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.environ.get('TORCHSIM_DIR', default='/root/workspace/PyTorchSim'))
 
    from Scheduler.scheduler import ExecutionEngine
    module = ExecutionEngine.setup_device()
    device = module.custom_device()
    test_sparse_mm(device, 512,512,512)#64, 64, 64)
    # test_sparse_mm("cpu", 128, 64, 32)