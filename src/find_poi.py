import argparse, os, sys, numpy as np
sys.path.append(os.path.dirname(__file__))
from data import TraceDataset
from utils import standardize

def nicv(traces, labels, num_classes=9):
    # traces: [N, T], labels: [N], classes 0..8 (HW)
    X = traces.astype(np.float32)
    X = standardize(X)  # per-trace z-score
    N, T = X.shape
    var_all = X.var(axis=0) + 1e-9

    means = np.zeros((num_classes, T), dtype=np.float64)
    probs = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        idx = (labels == c)
        if idx.any():
            probs[c] = idx.mean()
            means[c] = X[idx].mean(axis=0)
    mu = (probs[:, None] * means).sum(axis=0)
    var_condexp = (probs[:, None] * (means - mu)**2).sum(axis=0)
    return (var_condexp / var_all).astype(np.float32)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="profiling NPZ (SBOX-HW labels)")
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()
    Z = np.load(args.data, allow_pickle=True)
    traces = Z["traces"]; labels = Z["labels"]
    s = nicv(traces, labels)
    order = np.argsort(-s)  # descending
    top = order[:args.top]
    print("Trace length:", traces.shape[1])
    print("Top NICV sample idx:", top.tolist())
    print("Suggested poi_center:", int(np.median(top)))
    print("NICV@top:", [float(s[i]) for i in top])
