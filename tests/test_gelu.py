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

def test_GeLU(device, size=(128, 128), approximate='none'):
    torch.manual_seed(0)
    input = torch.randn(size)
    x1 = input.to(device=device)
    x2 = input.to("cpu")
    GeLU = torch.nn.GELU(approximate=approximate)
    opt_fn = torch.compile(dynamic=False)(GeLU)
    y = opt_fn(x1)
    cpu_y = GeLU(x2)
    test_result("GeLU", y, cpu_y)

if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim'))

    from Scheduler.scheduler import ExecutionEngine
    module = ExecutionEngine.setup_device()
    device = module.custom_device()
    # test_GeLU(device, (128, 128))
    # test_GeLU(device, (128, 128), approximate='tanh')
