import time
import torch
import torch.nn as nn
import torch_directml

dml = torch_directml.device()
print("torch:", torch.__version__)
print("device:", torch_directml.device_name(0))

def run(device, label):
    torch.manual_seed(0)
    # Conv block (CNN part): B x 3 x 32 x 32
    conv = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
    ).to(device)
    # Attention block (Transformer part): B x tokens x dim
    attn = nn.TransformerEncoderLayer(
        d_model=64, nhead=4, dim_feedforward=128, batch_first=True
    ).to(device)
    head = nn.Linear(64, 10).to(device)

    params = list(conv.parameters()) + list(attn.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3, foreach= False)

    x = torch.randn(128, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (128,), device=device)

    def step():
        f = conv(x)                              # B x 64 x 16 x 16
        t = f.flatten(2).transpose(1, 2)         # B x 256 x 64
        t = attn(t)                              # B x 256 x 64
        logits = head(t.mean(dim=1))             # B x 10
        loss = nn.functional.cross_entropy(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
        return loss.item()

    step()  # warm-up
    t0 = time.time()
    for _ in range(20):
        loss = step()
    print(f"{label}: loss={loss:.4f}  time/20 steps={time.time()-t0:.2f}s")

run(torch.device("cpu"), "CPU")
run(dml, "DirectML GPU")