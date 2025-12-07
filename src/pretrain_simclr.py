import argparse, yaml, os, sys
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# allow "python src/pretrain_simclr.py" without installing the package
sys.path.append(os.path.dirname(__file__))

from data import TraceDataset, AugParams
from models import SCAEncoder, ProjectionMLP
from utils import set_seed, pick_device

def info_nce(z1, z2, temp=0.1):
    z1 = F.normalize(z1, dim=1); z2 = F.normalize(z2, dim=1)
    N = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.t()) / temp
    mask = torch.eye(2*N, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, -1e9)
    labels = torch.arange(N, device=z.device)
    labels = torch.cat([labels+N, labels], dim=0)
    loss = F.cross_entropy(sim, labels)
    return loss

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--out", default="runs/enc_simclr.pt")
    args = ap.parse_args()

    # Load config
    cfg = yaml.safe_load(open(args.config))
    if args.epochs is not None:
        cfg["train"]["epochs"] = int(args.epochs)

    # --- Coerce numeric types in case YAML strings slipped in ---
    cfg["train"]["lr"] = float(cfg["train"]["lr"])
    cfg["train"]["weight_decay"] = float(cfg["train"]["weight_decay"])
    cfg["train"]["batch_size"] = int(cfg["train"]["batch_size"])
    cfg["simclr"]["temperature"] = float(cfg["simclr"]["temperature"])
    cfg["augment"]["time_shift"] = int(cfg["augment"]["time_shift"])
    cfg["augment"]["amp_scale"] = float(cfg["augment"]["amp_scale"])
    cfg["augment"]["add_noise_snr_db"] = float(cfg["augment"]["add_noise_snr_db"])

    set_seed(int(cfg["seed"]))
    device = pick_device(cfg.get("device", "auto"))

    # Data
    aug = AugParams(**cfg["augment"])
    ds = TraceDataset(
        args.data,
        poi_center=cfg["data"]["poi_center"],
        poi_width=int(cfg["data"]["poi_width"]),
        attacked_byte=int(cfg["data"]["attacked_byte"]),
        augment=aug
    )
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True, drop_last=True)

    # Model
    emb_dim = int(cfg["simclr"]["proj_dim"])
    enc = SCAEncoder(emb_dim=emb_dim).to(device)
    proj = ProjectionMLP(emb_dim=emb_dim, proj_dim=emb_dim).to(device)

    opt = optim.AdamW(
        list(enc.parameters()) + list(proj.parameters()),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"]
    )

    # Train
    epochs = int(cfg["train"]["epochs"])
    for epoch in range(epochs):
        enc.train(); proj.train()
        pbar = tqdm(loader, desc=f"pretrain {epoch+1}/{epochs}")
        for x, _, _ in pbar:
            x = x.to(device)
            # two simple "views": original batch and a shuffled view
            z1 = proj(enc(x))
            z2 = proj(enc(x[torch.randperm(x.size(0))]))
            loss = info_nce(z1, z2, temp=cfg["simclr"]["temperature"])
            opt.zero_grad(); loss.backward(); opt.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    # Save encoder
    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)
    torch.save(enc.state_dict(), args.out)
    print(f"[OK] saved encoder to {args.out}")

if __name__ == "__main__":
    main()
