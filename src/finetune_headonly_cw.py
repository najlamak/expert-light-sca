import argparse, yaml, os, sys, random, numpy as np, torch, torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Subset
sys.path.append(os.path.dirname(__file__))
from data import TraceDataset
from models import SCAEncoder, HWHead
from utils import set_seed, pick_device

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="runs/poi_153w250.yaml")
ap.add_argument("--data", required=True)
ap.add_argument("--encoder", required=True)
ap.add_argument("--shots", type=int, default=400)
ap.add_argument("--epochs", type=int, default=12)
ap.add_argument("--lr", type=float, default=0.001)
ap.add_argument("--out", default="runs/linear_headonly_cw.pt")
args = ap.parse_args()

cfg = yaml.safe_load(open(args.config))
set_seed(int(cfg["seed"])); device = pick_device(cfg.get("device","auto"))

ds = TraceDataset(args.data,
                  poi_center=cfg["data"]["poi_center"],
                  poi_width=int(cfg["data"]["poi_width"]),
                  attacked_byte=int(cfg["data"]["attacked_byte"]),
                  augment=None, standardize_input=True)

# per-class few-shot subset (no fancy sampler)
idx_by_cls = {c: [] for c in range(9)}
for i in range(len(ds)):
    _, y, _ = ds[i]; idx_by_cls[int(y)].append(i)
few_idx = []
for c, lst in idx_by_cls.items():
    random.shuffle(lst)
    few_idx += lst[:min(len(lst), args.shots)]

subset = Subset(ds, few_idx)
loader  = DataLoader(subset, batch_size=int(cfg["train"]["batch_size"]), shuffle=True, drop_last=True)

# class weights from the subset (inverse freq)
counts = np.bincount([int(ds[i][1]) for i in few_idx], minlength=9)
total  = counts.sum()
class_weights = torch.tensor([total/max(1,c) for c in counts], dtype=torch.float32).to(device)

enc = SCAEncoder().to(device); enc.load_state_dict(torch.load(args.encoder, map_location=device), strict=False)
for p in enc.parameters(): p.requires_grad = False  # head-only
head = HWHead().to(device)

opt = optim.AdamW(head.parameters(), lr=args.lr, weight_decay=5e-5)

for ep in range(1, args.epochs+1):
    enc.train(False); head.train(True)
    tot=0; correct=0; seen=0
    for x,y,_ in loader:
        x,y = x.to(device), y.to(device)
        with torch.no_grad(): z = enc(x)
        logits = head(z)
        loss = F.cross_entropy(logits, y, weight=class_weights)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item()*x.size(0)
        correct += (logits.argmax(1)==y).sum().item(); seen+=x.size(0)
    print(f"epoch {ep}: loss={tot/max(1,seen):.4f} acc={correct/max(1,seen):.3f}")

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
torch.save({"head": head.state_dict()}, args.out)
print("[OK] saved", args.out)
