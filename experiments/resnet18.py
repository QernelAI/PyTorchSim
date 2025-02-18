import torch
import torch._dynamo
import torch.utils.cpp_extension

import argparse
import subprocess
import datetime

def run_resnet(device, batch):
    from torchvision.models import resnet
    model = resnet._resnet(resnet.BasicBlock, [1, 1, 0, 0], weights=None, progress=False).eval()
    model.to(device, memory_format=torch.channels_last)
    input = torch.randn(batch, 3, 224, 224).to(device=device)
    x1 = input.to(device=device, memory_format=torch.channels_last)
    opt_fn = torch.compile(dynamic=False)(model)
    res = opt_fn(x1)
    print("ResNet18 Simulation Done")

if __name__ == "__main__":
    import os
    import sys
    base_dir = os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim')
    sys.path.append(base_dir)
    args = argparse.ArgumentParser()
    args.add_argument('--batch', type=int, default=1)
    args.add_argument('--dump_path', type=str, default='results')
    args = args.parse_args()
    batch = args.batch
    result_path = os.path.join(base_dir, args.dump_path, f"resnet18_{batch}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    # setting environment variables
    os.environ['TORCHSIM_DUMP_PATH'] = result_path
    # only timing simulation
    os.environ['TORCHSIM_VALIDATION_MODE'] = "0"
    if 'BACKENDSIM_SPIKE_ONLY' in os.environ:
        del os.environ['BACKENDSIM_SPIKE_ONLY']

    from Scheduler.scheduler import ExecutionEngine
    module = ExecutionEngine.setup_device()
    device = module.custom_device()
    run_resnet(device, batch)
    # compute cycles with shell script
    subprocess.run([f"{base_dir}/scripts/end2end.sh {result_path}"], shell=True)
