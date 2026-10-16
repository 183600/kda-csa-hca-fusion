#!/bin/bash
# AutoDL one-click script for the supported LM training entry point.

set -euo pipefail

echo "=== Setup env ==="
# Install the project dependencies plus the LM-training extras.
# requirements.txt must succeed first; the HF stack is then installed
# explicitly from the pinned ranges below.
pip install -r requirements.txt -q
pip install 'transformers>=4.36,<5' 'datasets>=2.14,<3' 'accelerate>=0.24,<1' 'tqdm>=4.60,<5' -q

echo "=== Show GPU ==="
nvidia-smi || true

echo "=== Train KDA+CSA+HCA Hybrid LM (1024 context) ==="
python train_lm_autodl.py --autodl --max_steps 2000 --seq_len 1024 --batch_size 2 --output_dir ./checkpoints_lm

echo "Done. Checkpoint: ./checkpoints_lm/final_lm.pt"
