import torch
import torch._dynamo
import torch.utils.cpp_extension
from torchvision.models import resnet18

def test_result(name, out, cpu_out, rtol=1e-4, atol=1e-4):
    if torch.allclose(out.cpu(), cpu_out, rtol=rtol, atol=atol):
        message = f"|{name} Test Passed|"
        print("-" * len(message))
        print(message)
        print("-" * len(message))
    else:
        message = f"|{name} Test Failed|"
        print("-" * len(message))
        print(message)
        print("-" * len(message))
        print("custom out: ", out.cpu())
        print("cpu out: ", cpu_out)
        exit(1)

def test_resnet(device):
    from torchvision.models import resnet
    # model = resnet._resnet(resnet.BasicBlock, [1, 1, 0, 0], weights=None, progress=False).eval()
    with torch.no_grad():
        model = resnet18().eval()
        model.to(device, memory_format=torch.channels_last)
        input = torch.randn(1, 3, 224, 224)
        x1 = input.to(device=device, memory_format=torch.channels_last)
        x2 = input.cpu().to(memory_format=torch.channels_last)
        opt_fn = torch.compile(dynamic=False)(model)
        res = opt_fn(x1)
        cpu_model = model.cpu().to(memory_format=torch.channels_last)
        cpu_res = cpu_model(x2)
    test_result("ResNet18 inference", res, cpu_res)
    print("Max diff > ", torch.max(torch.abs(res.cpu() - cpu_res)))
    print("ResNet18 Simulation Done")

if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim'))

    from Scheduler.scheduler import ExecutionEngine
    module = ExecutionEngine.setup_device()
    device = module.custom_device()
    test_resnet(device)
