from Scheduler.scheduler import Scheduler, SchedulerDNNModel, Request
import torch

def custom_matmul(a, b, c):
    #return torch.addmm(wieght2_, a, b)
    return torch.matmul(torch.matmul(a, b), c)

# Register compiled model
opt_fn = torch.compile()(custom_matmul)
SchedulerDNNModel.register_model("test_matmul", opt_fn)

# Init scheduler
scheduler = Scheduler(num_request_queue=2, engine_select=Scheduler.RR_ENGINE)

# Init request
input1 = torch.randn(128, 64)
weight1 = torch.randn(64, 32)
wieght2_1 = torch.randn(32,128)
input2 = torch.randn(128, 64)
weight2 = torch.randn(64, 32)
wieght2_2 = torch.randn(32,128)
input3 = torch.randn(128, 64)
weight3 = torch.randn(64, 32)
wieght2_3 = torch.randn(32,128)
input4 = torch.randn(128, 64)
weight4 = torch.randn(64, 32)
wieght2_4 = torch.randn(32,128)
input5 = torch.randn(128, 64)
weight5 = torch.randn(64, 32)
wieght2_5 = torch.randn(32,128)
input6 = torch.randn(128, 64)
weight6 = torch.randn(64, 32)
wieght2_6 = torch.randn(32,128)

new_request1 = Request("test_matmul", [input1], [weight1, wieght2_2], request_queue_idx=0)
new_request2 = Request("test_matmul", [input2], [weight2, wieght2_1], request_queue_idx=0)
new_request3 = Request("test_matmul", [input3], [weight3, wieght2_3], request_queue_idx=0)
new_request4 = Request("test_matmul", [input4], [weight4, wieght2_4], request_queue_idx=1)
new_request5 = Request("test_matmul", [input5], [weight5, wieght2_5], request_queue_idx=1)
new_request6 = Request("test_matmul", [input6], [weight6, wieght2_6], request_queue_idx=1)

# Add request to scheduler
scheduler.add_request(new_request1, 0)
scheduler.add_request(new_request2, 00)
scheduler.add_request(new_request3, 00)
scheduler.add_request(new_request4, 5000)
scheduler.add_request(new_request5, 15000)
scheduler.add_request(new_request6, 20000)

# Run scheduler
while not scheduler.is_finished():
    scheduler.schedule()
print("Done") 