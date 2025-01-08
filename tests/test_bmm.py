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

def test_BMM(device, batch_size=1, m=32, n=16, k=64):
    def bmm(a, b):
        return torch.bmm(a, b.transpose(1, 2))
    torch.manual_seed(0)
    a = torch.randn(batch_size, m, k).to(device=device)
    b = torch.randn(batch_size, n, k).to(device=device)
    opt_fn = torch.compile(dynamic=False)(bmm)
    res = opt_fn(a, b)
    out = bmm(a.cpu(), b.cpu())
    test_result("BMM Forward", res, out)

if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim'))

    from Scheduler.scheduler import ExecutionEngine
    module = ExecutionEngine.setup_device()
    device = module.custom_device()
    test_BMM(device)
    test_BMM(device, 2, 512, 512, 512)
