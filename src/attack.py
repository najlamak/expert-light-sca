import argparse, os, sys, yaml, numpy as np, torch
from torch.utils.data import DataLoader

# allow "python src/attack.py" without installing the package
sys.path.append(os.path.dirname(__file__))

from data import TraceDataset
from models import SCAEncoder, HWHead
from utils import AES_SBOX, HW, set_seed, pick_device

def hw_logits(encoder, head, loader, device):
    encoder.eval(); head.eval()
    outs = []; pts = []
    with torch.no_grad():
        for x,_,pt in loader:
            x = x.to(device)
            z = encoder(x)
            lo = head(z)              # logits over HW classes 0..8
            outs.append(lo.cpu())
            pts.extend([int(p) for p in pt])
    return torch.cat(outs), np.array(pts, dtype=np.uint8)

def score_keys(hw_logp, pts):
    """Sum log-probabilities over HW(SBOX(PT ^ k)) for each key guess k.
    Uses torch.gather for safe per-row indexing."""
    device = hw_logp.device
    Q = hw_logp.shape[0]
    scores = torch.zeros(256, device=device)
    for k in range(256):
        s = AES_SBOX[np.bitwise_xor(pts, k)]
        hw_idx = torch.as_tensor(HW[s], dtype=torch.long, device=device)  # (Q,)
        per_trace = hw_logp.gather(1, hw_idx.unsqueeze(1)).squeeze(1)     # (Q,)
        scores[k] = per_trace.sum()
    return scores

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--data", required=True, help=".npz with traces, labels, plaintexts")
    ap.add_argument("--encoder", required=True, help="path to encoder .pt (fallback)")
    ap.add_argument("--linear", required=True, help="few-shot bundle (may contain encoder+head) OR head-only .pt")
    ap.add_argument("--true-key", type=lambda x: int(x,0), default=None, help="e.g. 0x2b; omit if unknown")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    set_seed(int(cfg["seed"]))
    device = pick_device(cfg.get("device","auto"))

    # dataset (no augment for attack)
    ds = TraceDataset(args.data,
                      poi_center=cfg["data"]["poi_center"],
                      poi_width=int(cfg["data"]["poi_width"]),
                      attacked_byte=int(cfg["data"]["attacked_byte"]),
                      augment=None,
                      standardize_input=True)
    loader = DataLoader(ds, batch_size=256, shuffle=False)

    # models
    enc = SCAEncoder().to(device)
    # load fallback encoder first
    enc.load_state_dict(torch.load(args.encoder, map_location=device), strict=False)

    head = HWHead().to(device)
    # load bundle; if it contains encoder, override to ensure exact match with head
    w = torch.load(args.linear, map_location=device)
    if isinstance(w, dict) and "encoder" in w:
        enc.load_state_dict(w["encoder"], strict=False)
    if isinstance(w, dict) and "head" in w:
        head.load_state_dict(w["head"])
    else:
        # file directly contains head state_dict
        head.load_state_dict(w)

    # logits + key scoring
    logits, pts = hw_logits(enc, head, loader, device)
    logp = logits.log_softmax(dim=1)
    scores = score_keys(logp, pts)
    order = torch.argsort(scores, descending=True)

    if args.true_key is not None:
        rk = int((order == args.true_key).nonzero(as_tuple=True)[0])
        print(f"Key rank (all traces): {rk}")
    else:
        print("Top-5 key guesses (hex):", [hex(int(k)) for k in order[:5]])

if __name__ == "__main__":
    main()
