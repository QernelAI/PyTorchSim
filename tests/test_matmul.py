import torch
import torch._dynamo
import torch.utils.cpp_extension

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

def test_matmul(device, input_size=128, hidden_size=128, output_size=128):
    def custom_matmul(a, b):
        return torch.matmul(a, b)
    torch.manual_seed(0)
    input = torch.randn(input_size, hidden_size)
    weight = torch.randn(hidden_size, output_size)
    bias = torch.randn(output_size)
    x1 = input.to(device=device)
    w1 = weight.to(device=device)
    x2 = input.to("cpu")
    w2 = weight.to("cpu")
    opt_fn = torch.compile(dynamic=False)(custom_matmul)
    res = opt_fn(x1, w1)
    y = custom_matmul(x2, w2)
    test_result("Matmul Forward", res, y)

def test_addmm(device, input_size=128, hidden_size=128, output_size=128):
    def custom_matmul(bias, a, b):
        return torch.addmm(bias, a, b)
    torch.manual_seed(0)
    input = torch.randn(input_size, hidden_size)
    weight = torch.randn(hidden_size, output_size)
    bias = torch.randn(output_size)
    x1 = input.to(device=device)
    w1 = weight.to(device=device)
    b1 = bias.to(device=device)
    x2 = input.to("cpu")
    w2 = weight.to("cpu")
    b2 = bias.to("cpu")
    opt_fn = torch.compile(dynamic=False)(custom_matmul)
    res = opt_fn(b1, x1, w1)
    y = custom_matmul(b2, x2, w2)
    test_result("Addmm Forward", res, y)

if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.abspath("/workspace/PyTorchSim"))

    from Scheduler.scheduler import ExecutionEngine
    module = ExecutionEngine.setup_device()
    device = module.custom_device()
    test_matmul(device, 32, 32, 32)
    test_matmul(device, 128, 128, 128)
    test_matmul(device, 512, 512, 512)
    test_matmul(device, 129, 61, 56)
    test_addmm(device, 128, 128, 128)
    test_addmm(device, 129, 61, 56)
