import os
import sys
import torch
from torchvision.models import resnet18 as model1

sys.path.append(os.path.abspath("/workspace/PyTorchSim"))
from Scheduler.scheduler import Scheduler, SchedulerDNNModel, Request
from test_extension_backend import DecoderBlock as model2

target_model1 = model1().eval()
target_model2 = model2(768, 12).eval()

# Init scheduler
scheduler = Scheduler(num_request_queue=2, engine_select=Scheduler.FIFO_ENGINE)
# Register compiled model
opt_model1 = torch.compile(target_model1.to(device=scheduler.execution_engine.module.custom_device(), memory_format=torch.channels_last))
opt_model2 = torch.compile(target_model2.to(device=scheduler.execution_engine.module.custom_device()))
SchedulerDNNModel.register_model("resnet18", opt_model1)
SchedulerDNNModel.register_model("bert", opt_model2)

# Init input data
model_input1 = torch.randn(1, 3, 224, 224)
model_input2 = torch.randn(512, 768)

# Init request
new_request1 = Request("resnet18", [model_input1], [], request_queue_idx=0)
new_request2 = Request("bert", [model_input2], [], request_queue_idx=1)

# Add request to scheduler
scheduler.add_request(new_request1, request_time=0)
scheduler.add_request(new_request2, request_time=0)

# Run scheduler
while not scheduler.is_finished():
    scheduler.schedule()

print("Done")