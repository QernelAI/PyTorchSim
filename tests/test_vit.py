import torch
import torch._dynamo
import torch.utils.cpp_extension
import argparse
from torchvision import models
from torchvision.models.vision_transformer import _vision_transformer

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

def init_vit_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

    elif isinstance(m, torch.nn.LayerNorm):
        if m.weight is not None:
            torch.nn.init.ones_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

    elif isinstance(m, torch.nn.MultiheadAttention):
        # QKV projection
        if hasattr(m, 'in_proj_weight'):
            torch.nn.init.normal_(m.in_proj_weight, mean=0.0, std=0.02)
        if hasattr(m, 'in_proj_bias') and m.in_proj_bias is not None:
            torch.nn.init.normal_(m.in_proj_bias)

        # Output projection
        if hasattr(m, 'out_proj'):
            torch.nn.init.normal_(m.out_proj.weight, mean=0.0, std=0.02)
            if m.out_proj.bias is not None:
                torch.nn.init.normal_(m.out_proj.bias)

def test_vit(device, batch=1, shape=(3, 224, 224), num_layers=1, num_heads=12, hidden_dim=768, mlp_dim=3072):
    with torch.no_grad():
        #model = models.vit_b_16(models.ViT_B_16_Weights.IMAGENET1K_V1).eval()
        model = _vision_transformer(
            patch_size=16,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            weights=None,
            progress=False
        ).eval()
        model.apply(init_vit_weights)

        input_tensor = torch.randn(batch, *shape)
        x_device = input_tensor.to(device=device)
        x_cpu = input_tensor.cpu()

        model.to(device)
        opt_model = torch.compile(dynamic=False)(model)
        out_device = opt_model(x_device)

        cpu_model = model.cpu()
        out_cpu = cpu_model(x_cpu)

    test_result("VisionTransformer inference", out_device, out_cpu)
    print("Max diff > ", torch.max(torch.abs(out_device.cpu() - out_cpu)))
    print("VisionTransformer Simulation Done")

def test_multihead_attention(device, batch=1, seq_len=32, hidden_dim=768, num_heads=12):
    print(f"Testing MultiheadAttention (batch={batch}, seq_len={seq_len}, dim={hidden_dim}, heads={num_heads})")

    mha = torch.nn.MultiheadAttention(
        embed_dim=hidden_dim,
        num_heads=num_heads,
        batch_first=True,
        dropout=0.0,
    ).eval()
    mha.apply(init_vit_weights)

    x = torch.randn(seq_len, hidden_dim)
    query, key, value = x.clone(), x.clone(), x.clone()

    mha_device = mha.to(device)
    q1, k1, v1 = query.to(device), key.to(device), value.to(device)

    compiled_mha = torch.compile(mha_device, dynamic=False)
    with torch.no_grad():
        out_device, _ = compiled_mha(q1, k1, v1)

    mha_cpu = mha.cpu()
    q2, k2, v2 = query.cpu(), key.cpu(), value.cpu()
    with torch.no_grad():
        out_cpu, _ = mha_cpu(q2, k2, v2)

    test_result("MultiheadAttention", out_device, out_cpu)
    print("Max diff > ", torch.max(torch.abs(out_device.cpu() - out_cpu)))
    print("MultiheadAttention Simulation Done")

if __name__ == "__main__":
    import os
    import sys
    parser = argparse.ArgumentParser(description="Run Vision Transformer test with comparison")
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--shape', type=str, default="(3,224,224)", help="e.g. '(3,224,224)'")
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--hidden_dim', type=int, default=768)
    parser.add_argument('--mlp_dim', type=int, default=3072)
    args = parser.parse_args()

    shape = tuple(map(int, args.shape.strip('()').split(',')))

    sys.path.append(os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim'))
    from Scheduler.scheduler import ExecutionEngine
    module = ExecutionEngine.setup_device()
    device = module.custom_device()
    test_multihead_attention(device)
    #test_vit(
    #    device,
    #    batch=args.batch,
    #    shape=shape,
    #    num_layers=args.num_layers,
    #    num_heads=args.num_heads,
    #    hidden_dim=args.hidden_dim,
    #    mlp_dim=args.mlp_dim
    #)