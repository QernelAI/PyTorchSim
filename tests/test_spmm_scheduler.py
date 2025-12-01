import os
import sys
import torch
import argparse
sys.path.append(os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim'))
from Scheduler.scheduler import Scheduler, SchedulerDNNModel, Request
from test_sparse_core import SparseMLP as model1
from test_transformer import EncoderBlock as model2
CONFIG_TORCHSIM_DIR = os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--input_size", type=int, default=128, help="Input layer size")
    parser.add_argument("--hidden_size", type=int, default=128, help="Hidden layer size")
    parser.add_argument("--output_size", type=int, default=128, help="Output layer size")
    parser.add_argument("--w1_sparsity", type=float, default=0.5, help="Sparsity of first layer weights (0 to 1)")
    parser.add_argument("--w2_sparsity", type=float, default=0.5, help="Sparsity of second layer weights (0 to 1)")
    parser.add_argument("--config", type=str)
    args = parser.parse_args()

    batch_size = args.batch_size
    input_size = args.input_size
    hidden_size = args.hidden_size
    output_size = args.output_size
    w1_sparsity = args.w1_sparsity
    w2_sparsity = args.w2_sparsity
    config_path = f"{CONFIG_TORCHSIM_DIR}/TOGSim/configs/{args.config}"

    print("batch_size: ", batch_size)
    print("input_size: ", input_size)
    print("hidden_size: ", hidden_size)
    print("output_size: ", output_size)
    print("w1_sparsity: ", w1_sparsity)
    print("w2_sparsity: ", w2_sparsity)

    with torch.no_grad():
        # Init scheduler
        scheduler = Scheduler(num_request_queue=2, engine_select=Scheduler.FIFO_ENGINE,
                              togsim_config=config_path)

        target_model1 = model1(input_size, hidden_size, output_size, w1_sparsity, w2_sparsity, scheduler.execution_engine.module.custom_device()).eval()
        target_model2 = model2(768, 12).eval()

        # Register compiled model
        opt_model1 = torch.compile(target_model1.to(device=scheduler.execution_engine.module.custom_device()))
        opt_model2 = torch.compile(target_model2.to(device=scheduler.execution_engine.module.custom_device()))
        SchedulerDNNModel.register_model("mlp", opt_model1)
        SchedulerDNNModel.register_model("bert", opt_model2)

        # Init input data
        model_input1 = torch.randn(batch_size, input_size)
        model_input2 = torch.randn(1, 512, 768)

        # Init request
        new_request1 = Request("mlp", [model_input1], [], request_queue_idx=0)
        #new_request2 = Request("bert", [model_input2], [], request_queue_idx=1)


        # Add request to scheduler
        scheduler.add_request(new_request1, request_time=0)
        #scheduler.add_request(new_request2, request_time=0)

        # Run scheduler
        while not scheduler.is_finished():
            scheduler.schedule()