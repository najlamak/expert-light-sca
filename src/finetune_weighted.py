import argparse, yaml, os, sys, random, numpy as np, torch, torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__))
from data import TraceDataset
from models import SCAEncoder, HWHead
from utils import set_seed, pick_device

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/base.yaml")
ap.add_argument("--data", required=True)
ap.add_argument("--encoder", required=True)
ap.add_argument("--shots", type=int, default=500)
ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--out", default="runs/linear_weighted.pt")
args = ap.parse_args()

cfg = yaml.safe_load(open(args.config))
set_seed(int(cfg["seed"]))
device = pick_device(cfg.get("device","auto"))

ds = TraceDataset(args.data,
                  poi_center=cfg["data"]["poi_center"],
                  poi_width=int(cfg["data"]["poi_width"]),
                  attacked_byte=int(cfg["data"]["attacked_byte"]),
                  augment=None, standardize_input=True)

# per-class indices
idx_by_cls = {c: [] for c in range(9)}
for i in range(len(ds)):
    _, y, _ = ds[i]; idx_by_cls[int(y)].append(i)

# few-shot sample per class (handles scarcity)
few_idx = []
for c, lst in idx_by_cls.items():
    random.shuffle(lst)
    few_idx += lst[:min(len(lst), args.shots)]

if not few_idx: raise SystemExit("No few-shot samples selected; check labels/shots.")

# class weights (inverse frequency **on the few-shot subset**)
subset_labels = [int(ds[i][1]) for i in few_idx]
counts = np.bincount(subset_labels, minlength=9)
total = counts.sum()
class_weights = torch.tensor([total/max(1,c) for c in counts], dtype=torch.float32)

# weighted sampler to balance minibatches
weights_per_sample = [class_weights[int(ds[i][1])].item() for i in few_idx]
sampler = WeightedRandomSampler(weights_per_sample, num_samples=len(weights_per_sample), replacement=True)

loader = DataLoader(Subset(ds, few_idx),
                    batch_size=int(cfg["train"]["batch_size"]),
                    sampler=sampler, drop_last=True)

# model: encoder (frozen except last block) + HW head
enc = SCAEncoder().to(device)
enc.load_state_dict(torch.load(args.encoder, map_location=device), strict=False)
for p in enc.parameters(): p.requires_grad = False
for p in enc.blocks[-1].parameters(): p.requires_grad = True

head = HWHead().to(device)

opt = optim.AdamW([p for p in list(enc.parameters()) + list(head.parameters()) if p.requires_grad],
                  lr=0.005, weight_decay=5e-5)

for ep in range(1, args.epochs+1):
    enc.train(); head.train()
    tot=0.0; correct=0; seen=0
    pbar = tqdm(loader, desc=f"weighted {ep}/{args.epochs}")
    for x,y,_ in pbar:
        x,y = x.to(device), y.to(device)
        z = enc(x); logits = head(z)
        loss = F.cross_entropy(logits, y, weight=class_weights.to(device))
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item()*x.size(0)
        pred = logits.argmax(1)
        correct += (pred==y).sum().item(); seen += x.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/max(1,seen):.3f}")
    print(f"epoch {ep}: loss={tot/max(1,seen):.4f} acc={correct/max(1,seen):.3f}")

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
torch.save({"encoder": enc.state_dict(), "head": head.state_dict()}, args.out)
print(f"[OK] saved {args.out}")
