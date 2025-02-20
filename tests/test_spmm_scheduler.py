import os
import sys
import torch

args = sys.argv
# if len(args) == 6:
#     batch_size = int(args[1])
#     input_size = int(args[2])
#     hidden_size = int(args[3])
#     output_size = int(args[4])
#     w1_sparsity = int(args[5]) * 0.01
#     w2_sparsity = int(args[5]) * 0.01
# else:
#     print("Usage: python test_sparse_core.py <batch_size> <input_size> <hidden_size> <output_size> <bias_shift>")
#     exit(1)

batch_size = 16
input_size = 32
hidden_size = 64
output_size = 128
w1_sparsity = 0.1
w2_sparsity = 0.7

print("batch_size: ", batch_size)
print("input_size: ", input_size)
print("hidden_size: ", hidden_size)
print("output_size: ", output_size)
print("w1_sparsity: ", w1_sparsity)
print("w2_sparsity: ", w2_sparsity)

sys.path.append(os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim'))

from Scheduler.scheduler import Scheduler, SchedulerDNNModel, Request

from test_sparse_core import SparseMLP as model1
from test_transformer import DecoderBlock as model2

with torch.no_grad():


    # Init scheduler
    scheduler = Scheduler(num_request_queue=2, engine_select=Scheduler.FIFO_ENGINE,
                        backend_config="/root/workspace/PyTorchSim/PyTorchSimBackend/configs/heterogeneous_c2_simple_noc.json")

    target_model1 = model1(input_size, hidden_size, output_size, w1_sparsity, w2_sparsity, scheduler.execution_engine.module.custom_device()).eval()
    target_model2 = model2(768, 12).eval()

    # Register compiled model
    opt_model1 = torch.compile(target_model1.to(device=scheduler.execution_engine.module.custom_device()))
    opt_model2 = torch.compile(target_model2.to(device=scheduler.execution_engine.module.custom_device()))
    SchedulerDNNModel.register_model("mlp", opt_model1)
    SchedulerDNNModel.register_model("bert", opt_model2)

    # Init input data
    model_input1 = torch.randn(batch_size, input_size)
    model_input2 = torch.randn(512, 768)

    # Init request
    new_request1 = Request("mlp", [model_input1], [], request_queue_idx=1)
    new_request2 = Request("bert", [model_input2], [], request_queue_idx=0)


    # Add request to scheduler
    scheduler.add_request(new_request1, request_time=0)
    scheduler.add_request(new_request2, request_time=0)

    # Run scheduler
    while not scheduler.is_finished():
        scheduler.schedule()

print("Done")