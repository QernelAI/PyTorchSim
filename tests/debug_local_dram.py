"""Debug script for local DRAM mode — runs a single matmul without swallowing exceptions."""
import os
import sys

sys.path.append(os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim'))
os.environ.setdefault('TORCHSIM_DIR', '/workspace/PyTorchSim')

import torch
import torch._dynamo

from Scheduler.scheduler import PyTorchSimRunner
module = PyTorchSimRunner.setup_device()
device = module.custom_device()

print(f"TOGSIM_CONFIG = {os.environ.get('TOGSIM_CONFIG', '(not set, using default)')}")

torch._dynamo.reset()
a = torch.randn(1, 512).to(device=device)
b = torch.randn(512, 512).to(device=device)
fn = torch.compile(dynamic=False)(lambda a, b: torch.matmul(a, b))
print("Running matmul...")
result = fn(a, b)
print(f"Done. Result shape: {result.shape}")
