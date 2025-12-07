# Expert-Light SCA: SimCLR → Few-shot → Attack (ASCAD)

This repo reproduces the results of Applied Cryptography Subject in **“Toward Generic Side-Channel Attacks: An Expert-Light, Data-Efficient Deep Learning Pipeline”** by Najla Alkhater & Sara Alotaibi. 

## Quick start
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Convert ASCAD to NPZ
python tools/ascad_to_npz.py --h5 /path/to/ASCAD.h5 --out data/ascad_fixed.npz --dataset profiling
python tools/ascad_to_npz.py --h5 /path/to/ASCAD.h5 --out data/ascad_attack.npz --dataset attack
# Pretrain, adapt, attack (single-config)
PYTHONPATH=src python src/pretrain_simclr.py --data data/ascad_fixed.npz --epochs 5 --out runs/enc_simclr.pt
PYTHONPATH=src python src/finetune_linear.py --config runs/poi_153w250.yaml --data data/ascad_profile.npz --encoder runs/enc_simclr.pt --shots 200 --epochs 10 --out runs/linear_C153_W250_cfg_repro.pt
PYTHONPATH=src python src/attack.py --config runs/poi_153w250.yaml --data data/ascad_attack.npz --encoder runs/enc_simclr.pt --linear runs/linear_C153_W250_cfg_repro.pt --true-key 0x4d
