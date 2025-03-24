import torch
import copy
import torch._dynamo
import torch.utils.cpp_extension

def test_result(name, out, cpu_out, rtol=1e-4, atol=1e-4):
    message = f"|{name} Test Passed|"
    if torch.allclose(out.cpu(), cpu_out, rtol=rtol, atol=atol):
        print("-" * len(message))
        print(message)
        print("-" * len(message))
    else:
        print("custom out: ", out.cpu())
        print("cpu out: ", cpu_out)
        exit(1)

def test_indirect_vectoradd(device, size=(128, 128)):
    def vectoradd(a, idx, b):
        return a[idx] + b
    x = torch.randn(size, dtype=torch.float32).to(device=device)
    idx = torch.randint(0,128, [128]).to(device=device)
    y = torch.randn(128, dtype=torch.float32).to(device=device)
    opt_fn = torch.compile(dynamic=False)(vectoradd)
    res = opt_fn(x, idx, y)
    out = vectoradd(x.cpu(), idx.cpu(), y.cpu())
    test_result("VectorAdd", res, out)

def test_embedding(device, vocab_size, dim):
    emb = torch.nn.Embedding(vocab_size, dim)
    cpu_emb = copy.deepcopy(emb)

    prompt = torch.randint(0, 1023, [511], dtype=torch.int)
    cpu_prompt = copy.deepcopy(prompt)
    prompt = prompt.to(device=device)

    emb.to(device=device)
    opt_emb = torch.compile(dynamic=False)(emb)
    res = opt_emb(prompt)
    cpu_res = cpu_emb(cpu_prompt)
    test_result("Embedding", res, cpu_res)

if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim'))

    from Scheduler.scheduler import ExecutionEngine
    module = ExecutionEngine.setup_device()
    device = module.custom_device()
    test_indirect_vectoradd(device)
    #test_embedding(device, 1024, 2048)