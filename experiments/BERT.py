import torch
import torch._dynamo
import torch.utils.cpp_extension
import math
import copy

import argparse
import subprocess
import datetime

def test_result(name, out, cpu_out, rtol=1e-4, atol=1e-4):
    if torch.allclose(out.cpu(), cpu_out, rtol=rtol, atol=atol):
        message = f"|{name} Test Passed|"
        print("-" * len(message))
        print(message)
        print("-" * len(message))
    else:
        message = f"|{name} Test Failed|"
        print("-" * len(message))
        print(message)
        print("-" * len(message))
        print("custom out: ", out.cpu())
        print("cpu out: ", cpu_out)
        exit(1)

def clones(module, N):
    "Produce N identical layers."
    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class my_MultiheadAttention(torch.nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(my_MultiheadAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(torch.nn.Linear(d_model, d_model), 4)
        self.attn = None

    def forward(self, query, key, value):
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(-1, self.h, self.d_k).transpose(0, 1).contiguous()
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        p_attn = scores.softmax(dim=-1)
        x = torch.matmul(p_attn, value)
        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(0, 1)
            .contiguous()
            .view(-1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)

class DecoderBlock(torch.nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(DecoderBlock, self).__init__()
        self.multihead_attn = my_MultiheadAttention(num_heads, embed_dim)
        self.layer_norm = torch.nn.LayerNorm(embed_dim)
        self.ffn1 = torch.nn.Linear(embed_dim, embed_dim*4)
        self.act = torch.nn.ReLU()
        self.ffn2 = torch.nn.Linear(embed_dim*4, embed_dim)

    def forward(self, x):
        result = self.multihead_attn(x, x, x)
        result = self.layer_norm(result+x)

        ffn1_result = self.ffn1(result)
        act_result = self.act(ffn1_result)
        ffn2_result = self.ffn2(act_result)
        return self.layer_norm(ffn2_result + result)

def run_BERT(device, size, input_seq, validation):
    hidden_dim = {'base': 768, 'large': 1024, 'xlarge': 2048}
    embedding_size = {'base': 768, 'large': 1024, 'xlarge': 2048}
    heads = {'base': 12, 'large': 16, 'xlarge': 32} # hidden/64 https://arxiv.org/pdf/1909.11942
    cpu_query = torch.randn(input_seq, hidden_dim[size])
    decoder_block = DecoderBlock(embedding_size[size], heads[size])
    cpu_res = decoder_block(cpu_query)

    query = cpu_query.clone().to(device=device)
    decoder_block.to(device=device)
    opt_fn = torch.compile(dynamic=False)(decoder_block)
    res = opt_fn(query)

    if validation:
        test_result(f"BERT-{size} Forwrad", res, cpu_res)
    print(f"BERT-{size} Simulation Done")

if __name__ == "__main__":
    import os
    import sys
    base_dir = os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim')
    config = os.environ.get('TORCHSIM_CONFIG', default=f'{base_dir}/PyTorchSimBackend/configs/systolic_ws_128x128_c1_simple_noc_tpuv2.json')
    config = config.split('/')[-1].split('.')[0][9:] # extract config name from config path FIXME: gem5 result is different as directoy name
    sys.path.append(base_dir)
    args = argparse.ArgumentParser()
    args.add_argument('--size', type=str, default='base')
    args.add_argument('--dump_path', type=str, default='results')
    args.add_argument('--input_size', type=int, default=512)
    args.add_argument('--validation', type=int, default=0)
    args = args.parse_args()
    size = args.size
    input_seq = args.input_size
    result_path = os.path.join(base_dir, args.dump_path, config, f"BERT_{size}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    # setting environment variables
    os.environ['TORCHSIM_DUMP_PATH'] = result_path
    # only timing simulation
    os.environ['TORCHSIM_VALIDATION_MODE'] = str(args.validation)
    if 'BACKENDSIM_SPIKE_ONLY' in os.environ:
        del os.environ['BACKENDSIM_SPIKE_ONLY']

    from Scheduler.scheduler import ExecutionEngine
    module = ExecutionEngine.setup_device()
    device = module.custom_device()
    run_BERT(device, size, input_seq, args.validation)
    # compute cycles with shell script
    subprocess.run([f"{base_dir}/scripts/end2end.sh {result_path}"], shell=True)
