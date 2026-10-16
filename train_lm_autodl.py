"""
LM training for the KDA+CSA+HCA hybrid, built on ``ops_fused``.

The model uses ``HybridKCHAttention`` from ``ops_fused.py``, which already
implements:
- KDA with proper unit-norm q/k, g clamp, chunked path, conv lookback
- CSA with overlapped 2-branch compression, STE for indexer, sink logits
- HCA with heavy compression + dense + sliding window
- Decoding caches (ops_decoding_cache) for efficient long-context

Defaults: 2000 steps, d_model=256, 5 layers (3:1:1), seq 1024 — sized for a
small single-GPU run.
"""
import os, time, argparse, random, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
try:
    from tqdm import tqdm
except ImportError:  # tqdm is preinstalled on Kaggle/AutoDL but is
    # optional for a minimal local run — fall back to a no-op wrapper.
    def tqdm(*args, **kwargs):
        class _NoOpBar:
            def __init__(self, *a, **k):
                self.total = k.get("total", None)
                self.n = 0
            def update(self, n=1):
                self.n += n
            def close(self):
                pass
            def set_description(self, *a, **k):
                pass
            def set_postfix(self, *a, **k):
                # Called every optimizer step (``pbar.set_postfix``); must
                # exist or the no-tqdm fallback crashes the training loop.
                pass
            def __enter__(self):
                return self
            def __exit__(self, *a):
                self.close()
        return _NoOpBar(*args, **kwargs)

# import repo modules
from ops_fused import HybridConfig, HybridKCHAttention
from ops_kda_backend import validate_kda_backend
from kaggle_setup import configure_torch_for_device

# tokenizer / dataset handling (reuse TinyStories if available)
try:
    from transformers import AutoTokenizer
    from datasets import load_dataset
    HAS_HF = True
except Exception:
    # ImportError on hosts without the HF stack; also catches transient
    # import-time failures (e.g. a half-installed datasets). A bare
    # ``except:`` here would also swallow KeyboardInterrupt / SystemExit
    # during interpreter shutdown, so scope to Exception.
    HAS_HF = False


class _FallbackCharTokenizer:
    """Minimal char-level tokenizer used when ``transformers`` is absent.

    Keeps the training pipeline runnable with ZERO network / HuggingFace
    dependencies (e.g. a bare Kaggle CPU session without the optional
    ``pip install transformers datasets`` step, or an offline machine):
    the dataset falls back to synthetic text and this tokenizer maps
    printable ASCII characters to ids. Quality is obviously worse than a
    real BPE tokenizer — it exists only so the code path executes
    end-to-end; install ``transformers`` for the real TinyStories runs.
    """

    _SPECIALS = ["<pad>", "<unk>"]

    def __init__(self):
        chars = [chr(i) for i in range(32, 127)]  # printable ASCII
        self._itos = self._SPECIALS + chars
        self._stoi = {c: i for i, c in enumerate(self._itos)}
        self.pad_token_id = 0
        self.unk_token_id = 1

    def __len__(self):
        return len(self._itos)

    def encode(self, text, truncation=False, max_length=None):
        ids = [self._stoi.get(ch, self.unk_token_id) for ch in text]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return ids

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
        real_len = min(len(tokens), self.seq_len + 1)
        if real_len < self.seq_len + 1:
            tokens = tokens + [self.tokenizer.pad_token_id] * (self.seq_len + 1 - real_len)
        else:
            tokens = tokens[:self.seq_len + 1]
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        if real_len <= 1:
            labels[:] = -100
        elif real_len - 1 < labels.numel():
            labels[real_len - 1:] = -100
        return {"input_ids": input_ids, "labels": labels}

class LMWithHybrid(nn.Module):
    def __init__(self, vocab_size, cfg: HybridConfig):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, cfg.d_model)
        self.hybrid = HybridKCHAttention(cfg, total_layers=cfg.n_kda + cfg.n_csa + cfg.n_hca)
        self.norm_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None):
        x = self.embed(input_ids)
        x = self.hybrid(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   labels.reshape(-1), ignore_index=-100)
        return {"logits": logits, "loss": loss}

def count_params(m):
    seen = set()
    total = 0
    for p in m.parameters():
        if not p.requires_grad:
            continue
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
    return total


def run_decode_consistency_probe(model, device, seq_len, output_dir=None,
                                 seed=1234):
    """Prefill-decode consistency + decode-roofline probe of the trained LM.

    A trained checkpoint is only useful for generation if token-by-token
    decoding (KDA recurrent state + per-layer CSA/HCA incremental caches,
    ``run_decoding.HybridDecoding``) reproduces the full-sequence training
    forward. This runs both paths through the same weights on one synthetic
    batch (random token ids — the check is about numerical path equivalence,
    not language quality) and writes a JSON report next to the checkpoint.
    See ``lm_consistency`` for the accounting conventions.
    """
    import lm_consistency
    import cache_accounting

    T = max(8, min(int(seq_len), 64))
    prefill_len = max(4, T // 2)
    gen = torch.Generator(device='cpu').manual_seed(seed)
    ids = torch.randint(0, model.embed.num_embeddings, (1, T),
                        generator=gen).to(device)
    report = lm_consistency.run_lm_consistency_check(
        model, ids, prefill_len=prefill_len, device=device)
    # Reconcile the clone's live decode caches against the analytical
    # accounting (component-wise, both modelled and capacity-grown totals).
    report['cache_audit'] = cache_accounting.run_cache_audit_for_model(
        model, context_len=prefill_len, decode_steps=T - prefill_len,
        device=device, seed=seed)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, 'decode_consistency.json')
        tmp = path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, allow_nan=False)
        os.replace(tmp, path)
        report['report_path'] = path
    return report


def _print_consistency_report(report, title=''):
    roof = report.get('roofline') or {}
    print(f"[consistency]{' ' + title if title else ''} "
          f"max|dlogit|={report['max_abs_logit_diff']:.3e} "
          f"(decode-only {report['decode_max_abs_logit_diff']:.3e}), "
          f"top1_agreement={report['top1_agreement']:.4f}, "
          f"consistent={report['consistent']} "
          f"(tol={report['tolerance']:.1e}), "
          f"diagnosis={report.get('diagnosis', '?')}")
    selection = report.get('selection') or {}
    if selection.get('agreement') is not None:
        print(f"[consistency] CSA top-k selection agreement: "
              f"{selection['agreement']:.4f} "
              f"(decode {selection['decode_agreement']:.4f}, "
              f"{selection.get('n_compared', 0)} positions compared)")
        if report.get('diagnosis') == 'topk_selection_divergence':
            print("[consistency] NOTE: drift is explained by torch.topk "
                  "tie-breaking (see ops_decoding_cache docstring), not by "
                  "cache state corruption")
    if roof:
        print(f"[roofline] decode at T={roof['context_len']}: "
              f"AI={roof['arithmetic_intensity_flops_per_byte']:.3f} "
              f"FLOPs/byte, bound={roof.get('roofline_bound', '?')}, "
              f"decode={report['decode_ms_per_token']:.3f} ms/token, "
              f"{roof['bytes_per_token']:.0f} B/token")
    audit = report.get('cache_audit') or {}
    comp = audit.get('comparison')
    if comp:
        b = audit['bytes']
        print(f"[cache-audit] live==analytical at T={comp['context_len']}: "
              f"consistent={comp['consistent']} "
              f"(autograd clean={audit['autograd_hygiene']['clean']}, "
              f"bytes[{b['policy']}]={b['total_bytes']:.0f} vs "
              f"bf16={b['pure_bf16_total_bytes']:.0f})")
    if report.get('report_path'):
        print(f"[consistency] report saved: {report['report_path']}")


def run_checkpoint_consistency(path, device, seed=1234):
    """Load a ``train_lm_autodl`` checkpoint and probe prefill-decode."""
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        # torch < 2.0 has no ``weights_only`` kwarg.
        ckpt = torch.load(path, map_location='cpu')
    cfg = HybridConfig(**ckpt['cfg'])
    model = LMWithHybrid(int(ckpt['vocab']), cfg)
    model.load_state_dict(ckpt['model'])
    model.to(device).eval()
    return run_decode_consistency_probe(
        model, device, seq_len=64, output_dir=os.path.dirname(path) or '.',
        seed=seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle", action="store_true")
    parser.add_argument("--autodl", action="store_true")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="./checkpoints_lm")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for model init and DataLoader shuffle.")
    parser.add_argument("--consistency_check", action="store_true",
                        help="Load --checkpoint, run the prefill-decode "
                             "consistency + decode-roofline probe, then exit "
                             "without training.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint path for --consistency_check "
                             "(default: <output_dir>/final_lm.pt).")
    parser.add_argument("--skip_consistency_check", action="store_true",
                        help="Skip the automatic prefill-decode consistency "
                             "probe run after training finishes.")
    args = parser.parse_args()

    is_kaggle = args.kaggle or os.path.exists("/kaggle")
    info = configure_torch_for_device()
    device = info.device
    print(f"Device: {device}, is_kaggle={is_kaggle}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    if args.consistency_check:
        ckpt_path = args.checkpoint or os.path.join(
            args.output_dir, "final_lm.pt")
        if not os.path.exists(ckpt_path):
            raise SystemExit(f"checkpoint not found: {ckpt_path}")
        report = run_checkpoint_consistency(ckpt_path, device, seed=args.seed)
        _print_consistency_report(report, title=os.path.basename(ckpt_path))
        return

    if HAS_HF:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    else:
        print("[train_lm_autodl] transformers not installed; using the "
              "char-level fallback tokenizer over synthetic text "
              "(install `transformers datasets` for real TinyStories runs)")
        tokenizer = _FallbackCharTokenizer()
    if tokenizer is None:
        raise RuntimeError("Need transformers tokenizer")
    if getattr(tokenizer, "pad_token", None) is None and hasattr(tokenizer, "eos_token"):
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)

    kda_backend = validate_kda_backend(
        os.environ.get('KDA_BACKEND', 'reference')
    )
    if is_kaggle:
        cfg = HybridConfig(d_model=256, n_heads_qk=2, n_heads_v=2, head_dim_k=32, head_dim_v=32,
                           csa_m=8, csa_topk=4, hca_m2=16, n_kda=3, n_csa=1, n_hca=1,
                           kda_chunk_size=64, kda_backend=kda_backend)
        batch_size = 1
        grad_accum = 8
        seq_len = 512
        max_steps = 500
    else:
        cfg = HybridConfig(d_model=512, n_heads_qk=4, n_heads_v=4, head_dim_k=64, head_dim_v=64,
                           csa_m=16, csa_topk=8, hca_m2=64, n_kda=3, n_csa=1, n_hca=1,
                           kda_chunk_size=64, kda_backend=kda_backend)
        batch_size = 2
        grad_accum = 4
        seq_len = 1024
        max_steps = 2000

    if args.batch_size is not None:
        batch_size = args.batch_size
    if args.seq_len is not None:
        seq_len = args.seq_len
    if args.max_steps is not None:
        max_steps = args.max_steps

    print(f"Hybrid layout: {cfg.n_kda}:{cfg.n_csa}:{cfg.n_hca}, d={cfg.d_model}")
    print(f"Training profile: batch_size={batch_size}, grad_accum={grad_accum}, "
          f"seq_len={seq_len}, optimizer_steps={max_steps}, seed={args.seed}")
    model = LMWithHybrid(vocab_size, cfg).to(device)
    print(f"Params: {count_params(model)/1e6:.2f}M")
    print(f"Layout: {model.hybrid.layout_str()}")

    ds = TinyStoriesLM(tokenizer, seq_len=seq_len, max_samples=20000 if not is_kaggle else 5000)
    loader_gen = torch.Generator()
    loader_gen.manual_seed(args.seed + 1_000_000)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0,
                        generator=loader_gen)
    no_decay_ids = set()
    for submod in model.modules():
        if isinstance(submod, (nn.Embedding, nn.LayerNorm)):
            for p in submod.parameters(recurse=False):
                no_decay_ids.add(id(p))
    _POSITIONAL_BIAS_SUFFIXES = ('Ba', 'Bb', 'B_idx', 'B_pos')
    decay, no_decay = [], []
    seen = set()
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in seen:
            continue
        seen.add(id(p))
        leaf_name = name.rsplit('.', 1)[-1]
        if (id(p) in no_decay_ids or p.ndim <= 1
                or leaf_name in _POSITIONAL_BIAS_SUFFIXES):
            no_decay.append(p)
        else:
            decay.append(p)
    param_groups = [
        {'params': decay, 'weight_decay': 0.1},
        {'params': no_decay, 'weight_decay': 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=3e-4)

    use_amp = device.type == "cuda"
    if use_amp and torch.cuda.is_bf16_supported():
        autocast_dtype = torch.bfloat16
    elif use_amp:
        autocast_dtype = torch.float16
    else:
        autocast_dtype = torch.float32
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and autocast_dtype == torch.float16))
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
                try:
                    batch = next(it)
                except StopIteration:
                    break
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

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

        if accum_batches == 0:
            break

        if accum_batches < grad_accum:
            optimizer.zero_grad(set_to_none=True)
            continue

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(decay + no_decay, 1.0)
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
                        "optimizer_step": optimizer_step, "micro_step": micro_step,
                        "seed": args.seed},
                       os.path.join(args.output_dir, f"step_{optimizer_step}.pt"))

    pbar.close()
    elapsed = time.time() - start
    print(f"Finished in {elapsed/60:.2f} min")
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "vocab": vocab_size,
                "optimizer_step": optimizer_step, "micro_step": micro_step,
                "seed": args.seed},
               os.path.join(args.output_dir, "final_lm.pt"))

    if not args.skip_consistency_check:
        report = run_decode_consistency_probe(
            model, device, seq_len, args.output_dir, seed=args.seed)
        _print_consistency_report(report, title="final_lm.pt")

if __name__ == "__main__":
    main()
