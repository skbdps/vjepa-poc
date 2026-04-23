"""V-JEPA 2.1 ViT-L encoder forward pass on TPU with random weights."""
import torch
import torch_xla

print("Loading (encoder, predictor) tuple (random weights)...")
encoder, predictor = torch.hub.load(
    ".", "vjepa2_1_vit_large_384", source="local", pretrained=False
)
encoder.eval()
predictor.eval()

enc_params = sum(p.numel() for p in encoder.parameters())
pred_params = sum(p.numel() for p in predictor.parameters())
print(f"  encoder:   {enc_params/1e6:.1f}M params, class={encoder.__class__.__name__}")
print(f"  predictor: {pred_params/1e6:.1f}M params, class={predictor.__class__.__name__}")
print(f"  encoder children: {list(dict(encoder.named_children()).keys())[:8]}")

B, C, T, H, W = 1, 3, 16, 384, 384
dummy = torch.randn(B, C, T, H, W)
print(f"\nInput: {tuple(dummy.shape)}")

dev = torch_xla.device()
print(f"Moving encoder to {dev}...")
encoder = encoder.to(dev)
dummy = dummy.to(dev)

print("Encoder forward pass (XLA compile may take 1-3 min on first run)...")
with torch.no_grad():
    out = encoder(dummy)
    torch_xla.sync()

def describe(x, prefix=""):
    if isinstance(x, torch.Tensor):
        print(f"{prefix}Tensor shape={tuple(x.shape)} dtype={x.dtype} device={x.device}")
    elif isinstance(x, (list, tuple)):
        print(f"{prefix}{type(x).__name__} len={len(x)}")
        for i, v in enumerate(x):
            describe(v, prefix + f"  [{i}] ")
    elif isinstance(x, dict):
        print(f"{prefix}dict keys={list(x.keys())}")
        for k, v in x.items():
            describe(v, prefix + f"  [{k}] ")
    else:
        print(f"{prefix}{type(x).__name__}")

print("\nEncoder output:")
describe(out)
print("\nDone.")
