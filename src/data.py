from dataclasses import dataclass
import numpy as np, torch
from torch.utils.data import Dataset
from typing import Optional
from utils import standardize

@dataclass
class AugParams:
    time_shift: int = 20
    amp_scale: float = 0.1
    add_noise_snr_db: float = 25.0

def random_augment(x: np.ndarray, p: AugParams) -> np.ndarray:
    if p.time_shift and p.time_shift > 0:
        s = np.random.randint(-p.time_shift, p.time_shift+1)
        x = np.roll(x, s)
    if p.amp_scale and p.amp_scale > 0:
        x = x * (1.0 + np.random.uniform(-p.amp_scale, p.amp_scale))
    if p.add_noise_snr_db is not None:
        snr = 10**(p.add_noise_snr_db/10.0)
        sigp = np.mean(x**2) + 1e-12
        noisp = sigp / snr
        x = x + np.random.normal(0, np.sqrt(noisp), size=x.shape).astype(np.float32)
    return x

class TraceDataset(Dataset):
    def __init__(self, npz_path: str, poi_center: Optional[int]=None, poi_width: int=1000,
                 attacked_byte: int = 0, augment: Optional[AugParams]=None, standardize_input=True):
        Z = np.load(npz_path, allow_pickle=True)
        traces = Z['traces'].astype(np.float32)
        labels = Z['labels'].astype(np.uint8)
        pts = Z['plaintexts'].astype(np.uint8)
        if poi_center is None:
            poi_center = traces.shape[1] // 2
        L = poi_width
        start = max(0, poi_center - L//2)
        end = min(traces.shape[1], start + L)
        traces = traces[:, start:end]
        if standardize_input:
            traces = standardize(traces)
        self.traces = traces; self.labels = labels; self.plaintexts = pts; self.augment = augment

    def __len__(self): return self.traces.shape[0]

    def __getitem__(self, i):
        x = self.traces[i]
        if self.augment is not None:
            x = random_augment(x.copy(), self.augment)
        x = torch.tensor(x).float().unsqueeze(0)
        y = int(self.labels[i]); pt = int(self.plaintexts[i])
        return x, y, pt
