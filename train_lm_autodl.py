"""
LM Training for KDA+CSA+HCA Hybrid using the repo's ops_fused (superior)
Supports Kaggle and AutoDL with cost <120 CNY

This file is merged from the earlier toy training demo (/home/user/train.py)
but now uses the rigorous HybridKCHAttention from ops_fused.py which already
implements:
- KDA with proper unit-norm q/k, g clamp, chunked path, conv lookback
- CSA with overlapped 2-branch compression, STE for indexer, sink logits
- HCA with heavy compression + dense + sliding window
- Decoding caches (ops_decoding_cache) for efficient long-context

Cost control: default 2000 steps, d_model=256, 5 layers (3:1:1), seq 1024
3090/4090 cost remains well below 120 CNY for the default small run
"""
import os, time, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# import repo modules
from ops_fused import HybridConfig, HybridKCHAttention
from kaggle_setup import configure_torch_for_device

# tokenizer / dataset handling (reuse TinyStories if available)
try:
    from transformers import AutoTokenizer
    from datasets import load_dataset
    HAS_HF = True
except:
    HAS_HF = False

class TinyStoriesLM(Dataset):
    def __init__(self, tokenizer, seq_len=1024, max_samples=20000, split="train"):
        self.seq_len = seq_len
        if HAS_HF:
            try:
                ds = load_dataset("roneneldan/TinyStories", split=split)
                if max_samples:
                    ds = ds.select(range(min(len(ds), max_samples)))
                self.texts = ds["text"]
            except Exception as e:
                print(f"HF load failed {e}, using synthetic")
                self.texts = ["Once upon a time, " * 200 for _ in range(1000)]
        else:
            self.texts = ["Once upon a time, " * 200 for _ in range(1000)]
        self.tokenizer = tokenizer

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer.encode(text, truncation=True, max_length=self.seq_len + 1)
        # Keep the real encoded length BEFORE padding so the loss can ignore
        # padded target positions. GPT-2 has no native pad token, so callers
        # set pad_token=eos_token; checking token_id alone would also mask
        # genuine EOS targets. Length-based masking avoids that bias.
        real_len = min(len(tokens), self.seq_len + 1)
        if real_len < self.seq_len + 1:
            tokens = tokens + [self.tokenizer.pad_token_id] * (self.seq_len + 1 - real_len)
        else:
            tokens = tokens[:self.seq_len + 1]
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        # labels[j] predicts original token j+1. Positions >= real_len-1 are
        # padding targets and must not dominate the LM objective.
        if real_len <= 1:
            labels[:] = -100
        elif real_len - 1 < labels.numel():
            labels[real_len - 1:] = -100
        return {"input_ids": input_ids, "labels": labels}

class LMWithHybrid(nn.Module):
    def __init__(self, vocab_size, cfg: HybridConfig):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, cfg.d_model)
        self.hybrid = HybridKCHAttention(cfg, total_layers=cfg.n_kda + cfg.n_csa + cfg.n_hca or 5)
        self.norm_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(self, input_ids, labels=None):
        x = self.embed(input_ids)
        x = self.hybrid(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            # TinyStoriesLM already returns next-token labels aligned with each
            # input position (input_ids=tokens[:-1], labels=tokens[1:]). The
            # previous code sliced both tensors again (logits[:, :-1] vs
            # labels[:, 1:]), causing an off-by-one target: logits at position
            # t were trained to predict token t+2 instead of t+1, and the first
            # next-token target was silently dropped. Use the full aligned
            # tensors and let labels=-100 mask padded targets.
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   labels.reshape(-1), ignore_index=-100)
        return {"logits": logits, "loss": loss}

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle", action="store_true")
    parser.add_argument("--autodl", action="store_true")
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="./checkpoints_lm")
    args = parser.parse_args()

    is_kaggle = args.kaggle or os.path.exists("/kaggle")
    info = configure_torch_for_device()
    device = info.device
    print(f"Device: {device}, is_kaggle={is_kaggle}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2") if HAS_HF else None
    if tokenizer is None:
        raise RuntimeError("Need transformers tokenizer")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)

    # Config: rigorous version, but small for cost
    if is_kaggle:
        cfg = HybridConfig(d_model=256, n_heads_qk=2, n_heads_v=2, head_dim_k=32, head_dim_v=32,
                           csa_m=8, csa_topk=4, hca_m2=16, n_kda=3, n_csa=1, n_hca=1,
                           kda_chunk_size=64)
        batch_size = 1
        grad_accum = 8
        seq_len = 512
        max_steps = 500
    else:
        cfg = HybridConfig(d_model=512, n_heads_qk=4, n_heads_v=4, head_dim_k=64, head_dim_v=64,
                           csa_m=16, csa_topk=8, hca_m2=64, n_kda=3, n_csa=1, n_hca=1,
                           kda_chunk_size=64)
        batch_size = args.batch_size
        grad_accum = 4
        seq_len = args.seq_len
        max_steps = args.max_steps

    print(f"Hybrid layout: {cfg.n_kda}:{cfg.n_csa}:{cfg.n_hca}, d={cfg.d_model}")
    model = LMWithHybrid(vocab_size, cfg).to(device)
    print(f"Params: {count_params(model)/1e6:.2f}M")
    print(f"Layout: {model.hybrid.layout_str()}")

    ds = TinyStoriesLM(tokenizer, seq_len=seq_len, max_samples=20000 if not is_kaggle else 5000)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

    # Mixed precision policy. Kaggle T4 (sm_75) does NOT support BF16; forcing
    # ``dtype=torch.bfloat16`` there can crash or silently fall back to slow
    # emulation, invalidating the advertised Kaggle/AutoDL training results.
    # Use BF16 only when PyTorch reports support, otherwise use FP16 + GradScaler
    # on CUDA. CPU runs stay in fp32 (autocast disabled).
    use_amp = device.type == "cuda"
    if use_amp and torch.cuda.is_bf16_supported():
        autocast_dtype = torch.bfloat16
    elif use_amp:
        autocast_dtype = torch.float16
    else:
        autocast_dtype = torch.float32
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and autocast_dtype == torch.float16))
    print(f"AMP enabled={use_amp}, dtype={autocast_dtype}, grad_scaler={scaler.is_enabled()}")

    os.makedirs(args.output_dir, exist_ok=True)
    model.train()
    it = iter(loader)
    start = time.time()
    recent_step_losses = []
    optimizer_step = 0
    micro_step = 0
    pbar = tqdm(total=max_steps, desc="optimizer steps")
    optimizer.zero_grad(set_to_none=True)
    while optimizer_step < max_steps:
        accum_loss = 0.0
        accum_batches = 0
        for _ in range(grad_accum):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Reset recurrent state per micro-batch (fresh independent sequence).
            model.hybrid.reset_state()

            with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                                enabled=use_amp):
                out = model(input_ids, labels=labels)
                raw_loss = out["loss"]
            if raw_loss is None or not torch.isfinite(raw_loss):
                raise RuntimeError(
                    f"non-finite LM loss at optimizer_step={optimizer_step}, "
                    f"micro_step={micro_step}: {raw_loss}")
            scaled_loss = raw_loss / grad_accum
            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            accum_loss += float(raw_loss.detach().item())
            accum_batches += 1
            micro_step += 1

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        optimizer_step += 1
        step_loss = accum_loss / max(1, accum_batches)
        recent_step_losses.append(step_loss)
        pbar.update(1)
        pbar.set_postfix({"loss": f"{step_loss:.4f}"})

        if optimizer_step % 10 == 0:
            print(f"step {optimizer_step} loss {sum(recent_step_losses) / len(recent_step_losses):.4f}")
            recent_step_losses.clear()

        if optimizer_step % 500 == 0:
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "vocab": vocab_size,
                        "optimizer_step": optimizer_step, "micro_step": micro_step},
                       os.path.join(args.output_dir, f"step_{optimizer_step}.pt"))

    pbar.close()
    elapsed = time.time() - start
    print(f"Finished in {elapsed/60:.2f} min")
    cost_3090 = elapsed/3600*1.8
    cost_4090 = elapsed/3600*2.8
    print(f"Cost estimate 3090: {cost_3090:.2f} CNY, 4090: {cost_4090:.2f} CNY << 120 CNY")
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__,
                "optimizer_step": optimizer_step, "micro_step": micro_step},
               os.path.join(args.output_dir, "final_lm.pt"))

if __name__ == "__main__":
    main()
