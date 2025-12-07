import argparse, os, sys, yaml, numpy as np, torch
from torch.utils.data import DataLoader, Subset
sys.path.append(os.path.dirname(__file__))
from data import TraceDataset
from models import SCAEncoder, HWHead
from utils import AES_SBOX, HW, pick_device

def score_keys(hw_logp, pts):
    scores = torch.zeros(256, device=hw_logp.device)
    for k in range(256):
        s = AES_SBOX[np.bitwise_xor(pts, k)]
        idx = torch.as_tensor(HW[s], dtype=torch.long, device=hw_logp.device)
        scores[k] = hw_logp.gather(1, idx.unsqueeze(1)).squeeze(1).sum()
    return scores

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--encoder", required=True)
ap.add_argument("--linear", required=True)
ap.add_argument("--true-key", type=lambda x:int(x,0), required=True)
ap.add_argument("--Ns", nargs="+", type=int, default=[10,20,50,100,200,500,1000,2000,5000,10000])
ap.add_argument("--out", default="runs/kr_curve.csv")
args = ap.parse_args()

cfg = yaml.safe_load(open(args.config))
device = pick_device("auto")
ds = TraceDataset(args.data,
    poi_center=int(cfg["data"]["poi_center"]),
    poi_width=int(cfg["data"]["poi_width"]),
    attacked_byte=int(cfg["data"]["attacked_byte"]),
    augment=None, standardize_input=True)
ld = DataLoader(ds, batch_size=256, shuffle=False)

# load model (use encoder from bundle if present)
enc = SCAEncoder().to(device); head = HWHead().to(device)
wenc = torch.load(args.encoder, map_location=device)
enc.load_state_dict(wenc, strict=False)
wb = torch.load(args.linear, map_location=device)
if isinstance(wb, dict) and "encoder" in wb: enc.load_state_dict(wb["encoder"], strict=False)
if isinstance(wb, dict) and "head" in wb: head.load_state_dict(wb["head"])
else: head.load_state_dict(wb)
enc.eval(); head.eval()

# collect logits & plaintexts
outs=[]; pts=[]
with torch.no_grad():
    for x,_,pt in ld:
        x=x.to(device)
        z=enc(x); lo=head(z)
        outs.append(lo.log_softmax(1).cpu()); pts+= [int(p) for p in pt]
logp = torch.cat(outs)            # [Q,9]
pts  = np.array(pts, dtype=np.uint8)

# step through Ns and compute key-rank
import csv
with open(args.out,"w",newline="") as f:
    w=csv.writer(f); w.writerow(["N","KR"])
    for N in args.Ns:
        scores = score_keys(logp[:N], pts[:N])
        order  = torch.argsort(scores, descending=True)
        rk = int((order==args.true_key).nonzero(as_tuple=True)[0])
        w.writerow([N, rk])
print("[OK] wrote", args.out)
