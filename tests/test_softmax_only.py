"""Isolate softmax-only compilation + simulation on Q32 CIM config."""
import os
import sys

sys.path.append(os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim'))

import torch
import torch._dynamo

from Scheduler.scheduler import PyTorchSimRunner
module = PyTorchSimRunner.setup_device()
device = module.custom_device()


def fn(x):
    return torch.softmax(x, dim=-1)


torch._dynamo.reset()
x = torch.randn(1, 512).to(device=device)
opt_fn = torch.compile(dynamic=False)(fn)
print("Starting softmax compile + sim...")
_ = opt_fn(x)
print("Done!")
