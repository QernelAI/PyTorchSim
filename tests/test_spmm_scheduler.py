import os
import sys
import torch

sys.path.append(os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim'))

from Scheduler.scheduler import Scheduler, SchedulerDNNModel, Request

from test_sparse_core import MLP as model1


target_model1 = model1(16, 16, 16).eval()

# Init scheduler
scheduler = Scheduler(num_request_queue=2, engine_select=Scheduler.FIFO_ENGINE,
                      backend_config="/workspace/PyTorchSim/PyTorchSimBackend/configs/stonne_c1_simple_noc_tpuv3.json")
# Register compiled model

opt_model1 = torch.compile(target_model1.to(device=scheduler.execution_engine.module.custom_device()))
SchedulerDNNModel.register_model("mlp", opt_model1)

# Init input data
model_input1 = torch.randn(16, 16)

# Init request
new_request1 = Request("mlp", [model_input1], [], request_queue_idx=0)


# Add request to scheduler
scheduler.add_request(new_request1, request_time=0)

# Run scheduler
while not scheduler.is_finished():
    scheduler.schedule()

print("Done")