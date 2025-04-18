import torch
import torch._dynamo
import torch.utils.cpp_extension

import argparse
import subprocess
import datetime

def run_resnet(device, batch):
    from torchvision.models import resnet18
    model = resnet18().eval()
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
    config = os.environ.get('TORCHSIM_CONFIG', default=f'{base_dir}/PyTorchSimBackend/configs/systolic_ws_128x128_c1_simple_noc_tpuv2.json')
    config = config.split('/')[-1].split('.')[0][9:] # extract config name from config path
    sys.path.append(base_dir)
    args = argparse.ArgumentParser()
    args.add_argument('--batch', type=int, default=1)
    args.add_argument('--dump_path', type=str, default='results')
    args = args.parse_args()
    batch = args.batch
    result_path = os.path.join(base_dir, args.dump_path, config, f"resnet18_{batch}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
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
    subprocess.run([f"{base_dir}/scripts/sim_time.sh {result_path}"], shell=True)
