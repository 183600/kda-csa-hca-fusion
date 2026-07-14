"""
Full experiment: CSA+HCA replacing MLA in Kimi Linear
Supports Kaggle and AutoDL with cost <120 CNY
- Kaggle: uses 2xT4, bf16, gradient checkpoint off (T4 doesn't support bf16 well)
- AutoDL: 3090/4090, bf16 + compile + flash kernels if available
Cost control: default 2000 steps, ~150M params, seq 1024, batch 2*4=8 -> ~16M tokens -> ~1-2h on 3090 (~2-4 CNY)
Increase to 8192 seq for long-context phase (phase 2) 500 steps extra

Usage:
  Kaggle: python train.py --kaggle
  AutoDL: python train.py --autodl --use_wandb False
"""
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
from config import ModelConfig, TrainConfig
from model.hybrid_model import KimiDeepSeekHybridModel, count_params
from dataset import TinyStoriesDataset, get_tokenizer
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle", action="store_true", help="Kaggle mode: smaller batch, fp16")
    parser.add_argument("--autodl", action="store_true", help="AutoDL mode: use bf16 + compile")
    parser.add_argument("--use_mla", action="store_true", help="Train baseline MLA model for comparison")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--use_wandb", type=str, default="auto", help="True/False/auto")
    parser.add_argument("--phase2_long", action="store_true", help="Enable second phase long context training 4096")
    return parser.parse_args()

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def train():
    args = parse_args()
    # Detect environment
    is_kaggle = args.kaggle or os.path.exists("/kaggle")
    is_autodl = args.autodl or os.path.exists("/root/autodl-tmp")

    model_cfg = ModelConfig()
    train_cfg = TrainConfig()

    if is_kaggle:
        # Kaggle T4 x2 has 16GB each, use smaller config
        train_cfg.batch_size = 1
        train_cfg.grad_accum = 8
        train_cfg.seq_len = 512
        train_cfg.max_steps = 1000
        train_cfg.use_bf16 = False  # T4 no bf16, use fp16 via autocast
        train_cfg.use_compile = False
        model_cfg.num_hidden_layers = 8
        model_cfg.hidden_size = 512
        model_cfg.intermediate_size = 1024
        model_cfg.csa_top_k = 32
        print("Detected Kaggle mode")
    elif is_autodl:
        train_cfg.batch_size = 2
        train_cfg.grad_accum = 4
        train_cfg.seq_len = 1024
        train_cfg.max_steps = 2000
        train_cfg.use_bf16 = True
        train_cfg.use_compile = True
        print("Detected AutoDL mode - cost optimized")
    # override
    if args.max_steps: train_cfg.max_steps = args.max_steps
    if args.seq_len: train_cfg.seq_len = args.seq_len
    if args.batch_size: train_cfg.batch_size = args.batch_size
    train_cfg.output_dir = args.output_dir

    tokenizer = get_tokenizer()
    # Adjust vocab size to tokenizer
    model_cfg.vocab_size = len(tokenizer)
    print(f"Tokenizer vocab size: {model_cfg.vocab_size}")

    model = KimiDeepSeekHybridModel(model_cfg, use_mla=args.use_mla)
    print(f"Model layer types: {model.get_layer_types()}")
    print(f"Params: {count_params(model)/1e6:.2f}M")

    device = get_device()
    model = model.to(device)

    if train_cfg.use_compile and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile...")
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"Compile failed: {e}")

    # Dataset
    max_samples = 10000 if is_kaggle else 20000
    train_ds = TinyStoriesDataset(tokenizer, seq_len=train_cfg.seq_len, split="train", max_samples=max_samples)
    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay, betas=(0.9,0.95))

    # scheduler
    def lr_schedule(step):
        if step < train_cfg.warmup_steps:
            return step / train_cfg.warmup_steps
        # cosine decay
        progress = (step - train_cfg.warmup_steps) / (train_cfg.max_steps - train_cfg.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress)) * 0.1 + 0.9 * 0.1  # decay to 10%

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    scaler = torch.cuda.amp.GradScaler(enabled=(not train_cfg.use_bf16 and device=="cuda"))

    # wandb
    use_wandb = False
    if args.use_wandb == "True" or (args.use_wandb=="auto" and is_autodl):
        try:
            import wandb
            wandb.init(project="kimi-deepseek-hybrid", config={**model_cfg.__dict__, **train_cfg.__dict__, "is_kaggle": is_kaggle})
            use_wandb = True
        except:
            use_wandb = False

    os.makedirs(train_cfg.output_dir, exist_ok=True)
    model.train()
    global_step = 0
    running_loss = 0
    start_time = time.time()

    iter_loader = iter(train_loader)
    pbar = tqdm(total=train_cfg.max_steps, desc="Training")
    while global_step < train_cfg.max_steps:
        try:
            batch = next(iter_loader)
        except StopIteration:
            iter_loader = iter(train_loader)
            batch = next(iter_loader)

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # autocast
        if train_cfg.use_bf16 and device=="cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(input_ids, labels=labels)
                loss = out["loss"] / train_cfg.grad_accum
        else:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device=="cuda")):
                out = model(input_ids, labels=labels)
                loss = out["loss"] / train_cfg.grad_accum

        scaler.scale(loss).backward()
        running_loss += loss.item()

        if (global_step + 1) % train_cfg.grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        if (global_step+1) % 10 == 0:
            pbar.set_postfix({"loss": running_loss/10, "lr": scheduler.get_last_lr()[0]})
            if use_wandb:
                wandb.log({"loss": running_loss/10, "lr": scheduler.get_last_lr()[0], "step": global_step})
            running_loss = 0

        if (global_step+1) % train_cfg.eval_every == 0:
            # save quick eval perplexity estimate = exp(loss)
            ckpt_path = os.path.join(train_cfg.output_dir, f"step_{global_step+1}.pt")
            torch.save({"model": model.state_dict(), "config": model_cfg.__dict__, "step": global_step}, ckpt_path)
            print(f"Saved {ckpt_path}")

        global_step+=1
        pbar.update(1)
        if global_step>=train_cfg.max_steps:
            break

    pbar.close()
    # final save
    final_path = os.path.join(train_cfg.output_dir, "final.pt")
    torch.save({"model": model.state_dict(), "config": model_cfg.__dict__}, final_path)
    elapsed = time.time() - start_time
    print(f"Training finished in {elapsed/60:.2f} min")
    # cost estimate
    if is_autodl:
        # AutoDL 3090 ~1.8 CNY/h, 4090 ~2.8 CNY/h
        cost_3090 = elapsed/3600 * 1.8
        cost_4090 = elapsed/3600 * 2.8
        print(f"Estimated cost 3090: {cost_3090:.2f} CNY, 4090: {cost_4090:.2f} CNY (<120 CNY OK)")

    if args.phase2_long:
        print("Starting Phase 2: Long context adaptation 4096")
        # Increase rope and train with longer seq
        train_cfg.seq_len = 4096
        train_cfg.max_steps = global_step + 500
        train_ds_long = TinyStoriesDataset(tokenizer, seq_len=4096, split="train", max_samples=5000)
        train_loader_long = DataLoader(train_ds_long, batch_size=1, shuffle=True)
        iter_loader = iter(train_loader_long)
        pbar = tqdm(total=500, desc="Phase2 Long")
        for step in range(500):
            try:
                batch = next(iter_loader)
            except StopIteration:
                iter_loader = iter(train_loader_long)
                batch = next(iter_loader)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=train_cfg.use_bf16):
                out = model(input_ids, labels=labels)
                loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            if step % 10 ==0:
                pbar.set_postfix({"loss": loss.item()})
            pbar.update(1)
        pbar.close()
        torch.save({"model": model.state_dict(), "config": model_cfg.__dict__}, os.path.join(train_cfg.output_dir, "final_long.pt"))

if __name__ == "__main__":
    train()
