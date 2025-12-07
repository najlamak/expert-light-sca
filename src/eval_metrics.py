import argparse, os, sys, yaml, numpy as np, torch, matplotlib.pyplot as plt
from torch.utils.data import DataLoader
sys.path.append(os.path.dirname(__file__))
from data import TraceDataset, AugParams
from models import SCAEncoder, HWHead
from utils import AES_SBOX, HW, set_seed, pick_device

def logits_and_pts(enc, head, ds, device):
    loader = DataLoader(ds, batch_size=256, shuffle=False)
    outs, pts = [], []
    enc.eval(); head.eval()
    with torch.no_grad():
        for x,_,pt in loader:
            x = x.to(device)
            z = enc(x); lo = head(z)
            outs.append(lo.cpu()); pts.extend([int(p) for p in pt])
    return torch.cat(outs), np.array(pts, dtype=np.uint8)

def key_rank(hw_logits, pts, true_key):
    logp = hw_logits.log_softmax(dim=1)
    device = logp.device
    Q = hw_logits.shape[0]
    scores = torch.zeros(256, device=device)
    for k in range(256):
        s = AES_SBOX[np.bitwise_xor(pts, k)]
        hw_idx = torch.as_tensor(HW[s], dtype=torch.long, device=device)
        per_trace = logp.gather(1, hw_idx.unsqueeze(1)).squeeze(1)
        scores[k] = per_trace.sum()
    order = torch.argsort(scores, descending=True)
    rk = int((order==true_key).nonzero(as_tuple=True)[0])
    return rk

def sr_at_n(hw_logits, pts, true_key, Ns, trials=100, seed=1337):
    rng = np.random.default_rng(seed)
    logp = hw_logits.log_softmax(dim=1); device = logp.device
    Q = hw_logits.shape[0]; all_idx = np.arange(Q)
    res = []
    for N in Ns:
        ok = 0; ranks=[]
        for _ in range(trials):
            sel = rng.choice(all_idx, size=min(N,Q), replace=False)
            scores = torch.zeros(256, device=device)
            for k in range(256):
                s = AES_SBOX[np.bitwise_xor(pts[sel], k)]
                hw_idx = torch.as_tensor(HW[s], dtype=torch.long, device=device)
                per_trace = logp[sel].gather(1, hw_idx.unsqueeze(1)).squeeze(1)
                scores[k] = per_trace.sum()
            order = torch.argsort(scores, descending=True)
            rk = int((order==true_key).nonzero(as_tuple=True)[0])
            ranks.append(rk)
            if rk == 0: ok += 1
        res.append((N, ok/trials, float(np.mean(ranks)), float(np.std(ranks))))
    return res

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--data", required=True)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--linear", default=None)
    ap.add_argument("--head", default=None)
    ap.add_argument("--true-key", type=lambda x: int(x,0), required=True)
    ap.add_argument("--out", default="runs/eval.csv")
    ap.add_argument("--plot", default="runs/eval.png")
    ap.add_argument("--snr-db", type=float, default=None)
    ap.add_argument("--jitter", type=int, default=0)
    ap.add_argument("--trials", type=int, default=100)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    set_seed(int(cfg["seed"])); device = pick_device(cfg.get("device","auto"))

    aug = None
    if args.snr_db is not None or args.jitter:
        snr = args.snr_db if args.snr_db is not None else cfg["augment"]["add_noise_snr_db"]
        aug = AugParams(time_shift=args.jitter, amp_scale=0.0, add_noise_snr_db=snr)

    ds = TraceDataset(args.data,
                      poi_center=cfg["data"]["poi_center"],
                      poi_width=int(cfg["data"]["poi_width"]),
                      attacked_byte=int(cfg["data"]["attacked_byte"]),
                      augment=aug, standardize_input=True)

    enc = SCAEncoder().to(device); head = HWHead().to(device)
    enc.load_state_dict(torch.load(args.encoder, map_location=device), strict=False)
    if args.linear:
        w = torch.load(args.linear, map_location=device)
        head.load_state_dict(w["head"] if isinstance(w, dict) and "head" in w else w)
    else:
        head.load_state_dict(torch.load(args.head, map_location=device))

    lo, pts = logits_and_pts(enc, head, ds, device)
    rk = key_rank(lo, pts, args.true_key)
    print(f"[KR] all-traces rank = {rk}")

    Ns = cfg["attack"]["Ns"]
    rows = sr_at_n(lo, pts, args.true_key, Ns, trials=args.trials, seed=1337)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("N,SR,mean_rank,std_rank\n")
        for N, SR, mr, sr in rows:
            f.write(f"{N},{SR:.4f},{mr:.2f},{sr:.2f}\n")
    print(f"[OK] wrote {args.out}")

    import matplotlib
    matplotlib.use("Agg")
    plt.figure(figsize=(5,3.2))
    plt.plot([r[0] for r in rows], [r[1] for r in rows], marker="o")
    plt.xlabel("N attack traces"); plt.ylabel("Success rate (SR@N)")
    plt.ylim(0,1.0); plt.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(args.plot) or ".", exist_ok=True)
    plt.tight_layout(); plt.savefig(args.plot, dpi=160)
    print(f"[OK] wrote {args.plot}")
