import argparse, yaml, os, sys, random, torch, torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# allow "python src/finetune_linear.py" without installing the package
sys.path.append(os.path.dirname(__file__))

from data import TraceDataset
from models import SCAEncoder, HWHead
from utils import set_seed, pick_device

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--data", required=True)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--shots", type=int, default=50, help="per-class shots (HW classes 0..8)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--out", default="runs/linear.pt")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.epochs is not None:
        cfg.setdefault("linear", {})["epochs"] = int(args.epochs)

    # coerce numeric types (defensive)
    cfg["train"]["batch_size"] = int(cfg["train"]["batch_size"])
    cfg["linear"]["epochs"] = int(cfg["linear"]["epochs"])
    cfg["linear"]["lr"] = float(cfg["linear"]["lr"])
    cfg["linear"]["weight_decay"] = float(cfg["linear"]["weight_decay"])

    set_seed(int(cfg["seed"]))
    device = pick_device(cfg.get("device","auto"))

    # dataset (no augment for few-shot)
    ds = TraceDataset(
        args.data,
        poi_center=cfg["data"]["poi_center"],
        poi_width=int(cfg["data"]["poi_width"]),
        attacked_byte=int(cfg["data"]["attacked_byte"]),
        augment=None
    )

    # build class -> indices map (HW classes 0..8)
    idx_by_cls = {c: [] for c in range(9)}
    for i in range(len(ds)):
        _, y, _ = ds[i]
        idx_by_cls[int(y)].append(i)

    # compute class weights (inverse frequency)
    counts = {c: max(1,len(lst)) for c,lst in idx_by_cls.items()}
    total = sum(counts.values())
    import torch
    class_weights = torch.tensor([total/counts[c] for c in range(9)], dtype=torch.float32)

    # sample up to --shots per class (handle rare classes)
    few_idx = []
    for c, lst in idx_by_cls.items():
        random.shuffle(lst)
        few_idx += lst[:min(len(lst), args.shots)]

    if len(few_idx) == 0:
        raise RuntimeError("No samples selected for few-shot. Check your labels or --shots value.")

    loader = DataLoader(Subset(ds, few_idx), batch_size=cfg["train"]["batch_size"], shuffle=True, drop_last=True)

    # model: load encoder, freeze by default, train a small head
    enc = SCAEncoder().to(device)
    enc.load_state_dict(torch.load(args.encoder, map_location=device), strict=False)

    if not cfg["linear"].get("unfreeze_last_block", False):
        for p in enc.parameters(): p.requires_grad = False
    else:
        for p in enc.parameters(): p.requires_grad = False
        for p in enc.blocks[-1].parameters(): p.requires_grad = True  # tiny partial unfreeze

    head = HWHead().to(device)

    params = [p for p in list(enc.parameters()) + list(head.parameters()) if p.requires_grad]
    opt = optim.AdamW(params, lr=cfg["linear"]["lr"], weight_decay=cfg["linear"]["weight_decay"])

    # train head (and maybe last block)
    for ep in range(1, cfg["linear"]["epochs"]+1):
        enc.train(); head.train()
        pbar = tqdm(loader, desc=f"linear {ep}/{cfg['linear']['epochs']}")
        total = 0.0; correct = 0; n = 0
        for x,y,_ in pbar:
            x,y = x.to(device), y.to(device)
            z = enc(x)                 # frozen or partly unfrozen
            logits = head(z)           # 9-class HW logits
            loss = F.cross_entropy(logits, y, weight=class_weights.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * x.size(0)
            pred = logits.argmax(1)
            correct += (pred==y).sum().item()
            n += x.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/max(1,n):.3f}")
        print(f"epoch {ep}: loss={total/max(1,n):.4f}, acc={correct/max(1,n):.3f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"encoder": enc.state_dict(), "head": head.state_dict()}, args.out)
    print(f"[OK] saved few-shot bundle -> {args.out}")

if __name__ == "__main__":
    main()
