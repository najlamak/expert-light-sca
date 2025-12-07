import argparse, yaml, os, sys, torch, torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
sys.path.append(os.path.dirname(__file__))
from data import TraceDataset, AugParams
from models import SCAEncoder, HWHead
from utils import set_seed, pick_device

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/base.yaml")
ap.add_argument("--data", required=True)
ap.add_argument("--epochs", type=int, default=10)
ap.add_argument("--augment", action="store_true")
ap.add_argument("--out-prefix", default="runs/supervised")
args = ap.parse_args()

cfg = yaml.safe_load(open(args.config))
set_seed(int(cfg["seed"])); device = pick_device(cfg.get("device","auto"))
aug = AugParams(**cfg["augment"]) if args.augment else None
ds = TraceDataset(args.data, poi_center=cfg["data"]["poi_center"], poi_width=int(cfg["data"]["poi_width"]),
                  attacked_byte=int(cfg["data"]["attacked_byte"]), augment=aug)
n = len(ds); n_train = int(0.9*n); n_val = n-n_train
train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(0))
train_loader = DataLoader(train_ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=True, drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=512, shuffle=False)

enc = SCAEncoder().to(device); head = HWHead().to(device)
opt = optim.AdamW(list(enc.parameters())+list(head.parameters()), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["weight_decay"]))
for ep in range(1, args.epochs+1):
    enc.train(); head.train(); tot=0; correct=0; seen=0
    for x,y,_ in train_loader:
        x,y = x.to(device), y.to(device)
        z = enc(x); logits = head(z); loss = F.cross_entropy(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item()*x.size(0); correct += (logits.argmax(1)==y).sum().item(); seen+=x.size(0)
    enc.eval(); head.eval(); vtot=0; vcor=0; vseen=0
    with torch.no_grad():
        for x,y,_ in val_loader:
            x,y = x.to(device), y.to(device)
            z = enc(x); lo = head(z); vtot += F.cross_entropy(lo,y).item()*x.size(0)
            vcor += (lo.argmax(1)==y).sum().item(); vseen+=x.size(0)
    print(f"epoch {ep}: train_acc={correct/max(1,seen):.3f} val_acc={vcor/max(1,vseen):.3f}")

os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
torch.save(enc.state_dict(), args.out_prefix+"_enc.pt")
torch.save(head.state_dict(), args.out_prefix+"_head.pt")
print("[OK] saved", args.out_prefix+"_enc.pt", "and", args.out_prefix+"_head.pt")
