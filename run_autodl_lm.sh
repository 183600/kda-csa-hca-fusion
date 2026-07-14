#!/bin/bash
# AutoDL one-click script, cost <120 CNY
# Tested on PyTorch 2.2 base image with CUDA 12.1
# GPU: 3090(1.8/h) 2h~3.6 CNY, 4090(2.8/h) 2h~5.6 CNY

set -e

echo "=== Setup env ==="
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
pip install -r requirements.txt -q || pip install datasets transformers accelerate tqdm einops -q

# Try install fla-core for KDA speed, if fails fallback to pure PyTorch
pip install fla-core -q || echo "fla-core install failed, using pure pytorch fallback"

echo "=== Show GPU ==="
nvidia-smi

echo "=== Train Phase1 1024 context ==="
python train.py --autodl --max_steps 2000 --seq_len 1024 --batch_size 2 --output_dir ./checkpoints/csa_hca

echo "=== Train Baseline MLA for comparison (optional, quick 500 steps) ==="
python train.py --autodl --use_mla --max_steps 500 --seq_len 1024 --batch_size 2 --output_dir ./checkpoints/mla_baseline || echo "baseline skip"

echo "=== Evaluate ==="
python evaluate.py --ckpt ./checkpoints/csa_hca/final.pt --seq_len 1024

echo "=== Phase2 Long context adaptation 4096 (optional, +30min) ==="
python train.py --autodl --max_steps 500 --seq_len 2048 --batch_size 1 --output_dir ./checkpoints/csa_hca --phase2_long

echo "Done. Total cost should be <10 CNY on 3090"
