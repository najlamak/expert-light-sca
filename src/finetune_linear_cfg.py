import argparse, yaml, os, sys, random, numpy as np, torch, torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Subset

# make local imports work
sys.path.append(os.path.dirname(__file__))
from data import TraceDataset
from models import SCAEncoder, HWHead
from utils import set_seed, pick_device

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)            # YAML with data.poi_center/poi_width/etc.
    ap.add_argument("--data", required=True)              # NPZ (profiling split)
    ap.add_argument("--encoder", required=True)           # SimCLR encoder .pt
    ap.add_argument("--shots", type=int, default=200)     # per-class few-shot budget
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--out", required=True)               # output .pt (will include encoder+head)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    set_seed(int(cfg.get("seed", 1337)))
    device = pick_device(cfg.get("device", "auto"))

    # Use the SAME crop + standardization as eval
    ds = TraceDataset(
        args.data,
        poi_center=int(cfg["data"]["poi_center"]),
        poi_width=int(cfg["data"]["poi_width"]),
        attacked_byte=int(cfg["data"]["attacked_byte"]),
        augment=None,
        standardize_input=True,   # <<< important: match attack/eval
    )

    # Few-shot subset (up to --shots per HW class 0..8)
    idx_by_cls = {c: [] for c in range(9)}
    for i in range(len(ds)):
        _, y, _ = ds[i]
        idx_by_cls[int(y)].append(i)
    few_idx = []
    for c, lst in idx_by_cls.items():
        random.shuffle(lst)
        few_idx += lst[:min(len(lst), args.shots)]

    subset = Subset(ds, few_idx)
    loader  = DataLoader(subset, batch_size=int(cfg["train"]["batch_size"]), shuffle=True, drop_last=True)

    # Model: encoder + HW head
    enc = SCAEncoder().to(device)
    enc.load_state_dict(torch.load(args.encoder, map_location=device), strict=False)

    # Optional partial unfreeze via YAML: linear.unfreeze_last_block: true/false
    unfreeze = bool(cfg.get("linear", {}).get("unfreeze_last_block", False))
    for p in enc.parameters(): p.requires_grad = False
    if unfreeze and hasattr(enc, "blocks") and len(enc.blocks) > 0:
        for p in enc.blocks[-1].parameters(): p.requires_grad = True

    head = HWHead().to(device)

    lr = float(cfg.get("linear", {}).get("lr", 5e-3))
    wd = float(cfg.get("linear", {}).get("weight_decay", 5e-5))
    opt = optim.AdamW(
        [p for p in list(enc.parameters()) + list(head.parameters()) if p.requires_grad],
        lr=lr, weight_decay=wd
    )

    for ep in range(1, int(args.epochs) + 1):
        enc.train(); head.train()
        tot, correct, seen = 0.0, 0, 0
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            z = enc(x)
            logits = head(z)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            seen += x.size(0)
        print(f"epoch {ep}: loss={tot/max(1,seen):.4f} acc={correct/max(1,seen):.3f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"encoder": enc.state_dict(), "head": head.state_dict()}, args.out)
    print("[OK] saved", args.out)

if __name__ == "__main__":
    main()
