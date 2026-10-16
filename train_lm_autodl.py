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
import os, time, argparse, random, json, gc, hashlib, datetime
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
from kaggle_setup import (configure_torch_for_device, set_global_seed,
                          seed_worker, capture_provenance,
                          write_json_atomic, write_results_json)

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
        self.split = split
        self.max_samples = max_samples
        if HAS_HF:
            try:
                ds = load_dataset("roneneldan/TinyStories", split=split)
                if max_samples:
                    ds = ds.select(range(min(len(ds), max_samples)))
                self.texts = ds["text"]
                self.data_source = "hf:roneneldan/TinyStories"
            except Exception as e:
                print(f"HF load failed {e}, using synthetic")
                self.texts = ["Once upon a time, " * 200 for _ in range(1000)]
                self.data_source = "synthetic:once_upon_a_time (HF load failed)"
        else:
            self.texts = ["Once upon a time, " * 200 for _ in range(1000)]
            self.data_source = "synthetic:once_upon_a_time (no transformers)"
        self.tokenizer = tokenizer
        # Content fingerprint so a run record can identify WHICH data was
        # trained on. Uses a bounded, deterministic sample (first
        # ``n_probe`` texts truncated per-text) — hashing every byte of a
        # 20k-story corpus on every start is wasteful, and the sample is
        # enough to detect a swapped split / stale cache.
        self.data_fingerprint = _texts_fingerprint(self.texts)

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


def _texts_fingerprint(texts, n_probe=64, max_chars_per_text=4096):
    """Bounded content fingerprint of the training texts (data version id).

    Hashes a deterministic sample of at most ``n_probe`` evenly spaced
    texts (each truncated to ``max_chars_per_text``) plus the corpus size,
    so a run record can detect a swapped split / stale dataset cache
    without hashing gigabytes on every launch.
    """
    n = len(texts)
    h = hashlib.sha256()
    h.update(
        f"n={n};probe={min(n, n_probe)};max_chars={max_chars_per_text}".encode())
    if n:
        stride = max(1, n // max(1, n_probe))
        seen = 0
        for i in range(0, n, stride):
            if seen >= n_probe:
                break
            h.update(str(texts[i])[:max_chars_per_text].encode(
                'utf-8', 'replace'))
            seen += 1
        # Always include the last sample so a truncation of the corpus (e.g.
        # a shorter stale cache) changes the fingerprint even if the stride
        # sample happens to coincide.
        h.update(str(texts[-1])[:max_chars_per_text].encode(
            'utf-8', 'replace'))
    return h.hexdigest()


def _batch_fingerprint(batch):
    """sha256 over the fixed overfit batch (ids + labels + shapes)."""
    h = hashlib.sha256()
    for name in ("input_ids", "labels"):
        t = batch[name].detach().to('cpu')
        h.update(f"{name}:{tuple(t.shape)}:".encode())
        h.update(str(t.contiguous().view(-1).tolist()).encode())
    return h.hexdigest()


def run_overfit_sanity_check(model_factory, batch, device, *, steps=100,
                             lr=3e-3, seed=42, grad_clip=1.0,
                             loss_ratio_threshold=0.5, log_every=10):
    """Overfit ONE fixed batch as a gradient-flow sanity gate.

    Runs before full training and trains a FRESH model (``model_factory()``)
    on a single fixed batch with all regularization disabled
    (``weight_decay=0``; the hybrid stack has no dropout and this pipeline
    applies no data augmentation — raw token windows only). If a model
    cannot drive the loss down on a handful of examples, something is
    fundamentally broken (gradients not flowing, detached loss, wrong
    labels, optimizer not stepping) and starting a 2000-step run would just
    burn compute.

    Pass criterion: the mean loss over the last ``tail`` steps must be
    strictly below ``loss_ratio_threshold`` times the mean loss over the
    first ``warmup`` steps (default: at least a 50% reduction). The token
    accuracy trajectory is reported for inspection but is informative
    only: this is a NECESSARY, NOT SUFFICIENT check — data-augmentation,
    feature-alignment and label-mapping bugs can still pass it, so it must
    be complemented by inspecting intermediate results.

    Returns a JSON-serializable report dict with the loss/accuracy curves.
    """
    if isinstance(device, str):
        device = torch.device(device)
    steps = int(steps)
    if steps < 10:
        # A handful of steps is dominated by early-loss noise (the loss can
        # even rise for a few steps before collapsing); requiring >= 10
        # avoids a misleading FAIL on a healthy model.
        raise ValueError(
            f"overfit check needs steps >= 10 to be meaningful, got {steps}")
    warmup = max(1, min(5, steps // 10))
    tail = max(1, min(5, steps // 10))

    # Deterministic, regularization-free recipe. Re-seeding here makes the
    # probe model's init independent of any RNG consumption before the call.
    seed_info = set_global_seed(seed, deterministic=True)
    model = model_factory().to(device)
    model.train()
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

    losses, accs, elapsed_s = [], [], []
    t0 = time.time()
    for step in range(steps):
        model.hybrid.reset_state()
        out = model(input_ids, labels=labels)
        loss = out["loss"]
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError(
                f"overfit sanity check: non-finite LM loss at step {step} "
                f"({loss}); aborting before full training.")
        with torch.no_grad():
            mask = labels.ne(-100)
            correct = ((out["logits"].argmax(-1) == labels) & mask).sum()
            acc = float(correct) / max(1, int(mask.sum()))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        losses.append(float(loss.detach()))
        accs.append(acc)
        elapsed_s.append(time.time() - t0)
        if log_every and (step % log_every == 0 or step == steps - 1):
            print(f"[overfit-check] step {step:4d} loss {losses[-1]:.4f} "
                  f"token_acc {acc:.4f} ({elapsed_s[-1]:.1f}s)", flush=True)

    initial_loss = sum(losses[:warmup]) / warmup
    final_loss = sum(losses[-tail:]) / tail
    best_loss = min(losses)
    loss_reduction = (initial_loss - final_loss) / initial_loss
    final_acc = sum(accs[-tail:]) / tail
    passed = final_loss < loss_ratio_threshold * initial_loss
    report = {
        "passed": bool(passed),
        "steps": steps,
        "lr": float(lr),
        "weight_decay": 0.0,
        "grad_clip_norm": float(grad_clip),
        "dropout": 0.0,
        "data_augmentation": "none",
        "seed": int(seed),
        "seed_info": seed_info,
        "fixed_batch_sha256": _batch_fingerprint(batch),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "best_loss": best_loss,
        "loss_reduction": loss_reduction,
        "loss_ratio_threshold": float(loss_ratio_threshold),
        "final_train_token_acc": final_acc,
        "losses": losses,
        "accs": accs,
        "elapsed_s": elapsed_s,
        "note": ("Necessary but NOT sufficient: passing proves gradients flow "
                 "and the loss responds on a fixed batch; it cannot rule out "
                 "data-augmentation / feature-alignment / label-mapping bugs."),
    }
    del model, opt
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


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
    # Provenance so a saved report identifies the code/env/seed that made it.
    report['seed'] = int(seed)
    report['provenance'] = capture_provenance()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, 'decode_consistency.json')
        write_json_atomic(report, path, indent=2, allow_nan=False)
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
    parser.add_argument("--skip_overfit_check", action="store_true",
                        help="Skip the small-batch overfit sanity gate before "
                             "training (NOT recommended: the gate catches "
                             "broken gradient flow / loss wiring cheaply).")
    parser.add_argument("--overfit_steps", type=int, default=100,
                        help="Optimizer steps for the fixed-batch overfit "
                             "sanity gate (default: 100).")
    parser.add_argument("--overfit_lr", type=float, default=3e-3,
                        help="LR for the overfit gate. Deliberately higher "
                             "than the 3e-4 training LR so a healthy model "
                             "memorizes the batch quickly (default: 3e-3).")
    parser.add_argument("--overfit_loss_ratio", type=float, default=0.5,
                        help="Gate passes when the tail-mean loss is strictly "
                             "below this fraction of the head-mean loss "
                             "(default: 0.5).")
    parser.add_argument("--overfit_check_only", action="store_true",
                        help="Run ONLY the overfit sanity gate (no full "
                             "training); useful as a fast smoke test.")
    args = parser.parse_args()

    is_kaggle = args.kaggle or os.path.exists("/kaggle")
    info = configure_torch_for_device()
    device = info.device
    print(f"Device: {device}, is_kaggle={is_kaggle}")

    # Seed Python random / NumPy / torch / CUDA together; deterministic
    # cuDNN + algorithm selection for same-machine reproducibility.
    seed_info = set_global_seed(args.seed, deterministic=True)
    # Provenance (git commit / dirty flag / torch+cuda versions / device) is
    # captured ONCE at run start and embedded in every artifact this script
    # writes (checkpoints, run record, consistency report).
    provenance = capture_provenance()

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

    max_train_samples = 20000 if not is_kaggle else 5000
    ds = TinyStoriesLM(tokenizer, seq_len=seq_len,
                       max_samples=max_train_samples)
    print(f"Data: {ds.data_source}, n={len(ds)}, "
          f"fingerprint={ds.data_fingerprint[:16]}...")
    loader_gen = torch.Generator()
    loader_gen.manual_seed(args.seed + 1_000_000)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0,
                        generator=loader_gen, worker_init_fn=seed_worker)

    # ------------------------------------------------------------------
    # Overfit sanity gate (before any full training): train a FRESH copy of
    # the model on ONE fixed batch with regularization off. A model that
    # cannot memorize a handful of examples has a fundamental bug
    # (gradients not flowing, detached loss, wrong labels, optimizer not
    # stepping) and a 500-2000 step run would be wasted. This is necessary
    # but NOT sufficient — augmentation/alignment/label-map bugs can still
    # pass — so it complements, never replaces, intermediate-result checks.
    # ------------------------------------------------------------------
    overfit_report = None
    overfit_batch = None
    if not args.skip_overfit_check:
        # Separate generator + loader: the probe batch is fixed across runs
        # and consuming it does NOT advance the training loader's stream, so
        # training stays bit-identical to a run with the gate skipped.
        probe_gen = torch.Generator()
        probe_gen.manual_seed(args.seed + 2_000_000)
        probe_loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                                  num_workers=0, generator=probe_gen)
        overfit_batch = next(iter(probe_loader))
        print(f"[overfit-check] training {args.overfit_steps} steps on one "
              f"fixed batch of {int(overfit_batch['input_ids'].shape[0])} "
              f"sequence(s) (weight_decay=0, no dropout/augmentation)...")
        overfit_report = run_overfit_sanity_check(
            lambda: LMWithHybrid(vocab_size, cfg), overfit_batch, device,
            steps=args.overfit_steps, lr=args.overfit_lr, seed=args.seed,
            loss_ratio_threshold=args.overfit_loss_ratio)
        print(f"[overfit-check] result: {'PASS' if overfit_report['passed'] else 'FAIL'} "
              f"loss {overfit_report['initial_loss']:.3f} -> "
              f"{overfit_report['final_loss']:.3f} "
              f"(best {overfit_report['best_loss']:.3f}), "
              f"token_acc -> {overfit_report['final_train_token_acc']:.3f}")
        if not overfit_report['passed']:
            raise SystemExit(
                "[overfit-check] FAILED: the model could not overfit a single "
                f"fixed batch in {args.overfit_steps} steps "
                f"(loss {overfit_report['initial_loss']:.3f} -> "
                f"{overfit_report['final_loss']:.3f}; required < "
                f"{args.overfit_loss_ratio:.2f}x the initial loss). This "
                "usually means gradients are not flowing, the loss is "
                "mis-wired, or labels are wrong — refusing to start the full "
                "run. Debug with --overfit_check_only, or bypass (NOT "
                "recommended) with --skip_overfit_check.")
        if args.overfit_check_only:
            print("[overfit-check] --overfit_check_only set; exiting without "
                  "full training.")
            return

    # Re-seed right before the real model is constructed so its init is
    # identical to a run where the gate never ran.
    set_global_seed(args.seed, deterministic=True)
    model = LMWithHybrid(vocab_size, cfg).to(device)
    print(f"Params: {count_params(model)/1e6:.2f}M")
    print(f"Layout: {model.hybrid.layout_str()}")

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

    # Hyperparameters are captured ONCE here and embedded in every
    # checkpoint / run record so no artifact is ambiguous about its recipe.
    hyperparams = {
        'optimizer': 'AdamW',
        'lr': 3e-4,
        'weight_decay_decay_group': 0.1,
        'weight_decay_no_decay_group': 0.0,
        'grad_clip_norm': 1.0,
        'batch_size': batch_size,
        'grad_accum': grad_accum,
        'effective_batch_sequences': batch_size * grad_accum,
        'seq_len': seq_len,
        'max_optimizer_steps': max_steps,
        'amp_enabled': use_amp,
        'autocast_dtype': str(autocast_dtype),
        'kda_backend': kda_backend,
        'model_config': dict(cfg.__dict__),
    }
    data_record = {
        'source': ds.data_source,
        'split': ds.split,
        'max_samples_budget': max_train_samples,
        'n_samples': len(ds),
        'tokenizer': type(tokenizer).__name__,
        'vocab_size': vocab_size,
        'content_fingerprint_sha256': ds.data_fingerprint,
        'fingerprint_scheme': ('sha256 of <=64 evenly spaced texts truncated '
                               'to 4096 chars + corpus size'),
    }
    created_utc = datetime.datetime.now(
        datetime.timezone.utc).isoformat()

    os.makedirs(args.output_dir, exist_ok=True)
    model.train()
    it = iter(loader)
    start = time.time()
    recent_step_losses = []
    # Full per-optimizer-step training curve (recorded in checkpoints and in
    # the run record; "don't log experiments by hand" — the files ARE the log).
    curve_steps, curve_losses, curve_elapsed_s = [], [], []
    optimizer_step = 0
    micro_step = 0

    def _checkpoint_payload():
        """Self-describing checkpoint: weights + recipe + provenance + curve.

        Recording the hyperparameters, git revision (and dirty flag), seed
        and the loss history INSIDE each checkpoint means even a crashed run's
        partial checkpoint can be traced back to the exact code/data/recipe
        that produced it.
        """
        return {
            "model": model.state_dict(),
            "cfg": cfg.__dict__,
            "vocab": vocab_size,
            "optimizer_step": optimizer_step,
            "micro_step": micro_step,
            "seed": args.seed,
            "seed_info": seed_info,
            "created_utc": created_utc,
            "git_commit": provenance.get("git_commit"),
            "git_dirty": provenance.get("git_dirty"),
            "hyperparams": hyperparams,
            "data": data_record,
            "loss_history": list(curve_losses),
        }

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
        curve_steps.append(optimizer_step)
        curve_losses.append(step_loss)
        curve_elapsed_s.append(time.time() - start)
        pbar.update(1)
        pbar.set_postfix({"loss": f"{step_loss:.4f}"})

        if optimizer_step % 10 == 0:
            print(f"step {optimizer_step} loss {sum(recent_step_losses) / len(recent_step_losses):.4f}")
            recent_step_losses.clear()

        if optimizer_step % 500 == 0:
            torch.save(_checkpoint_payload(),
                       os.path.join(args.output_dir, f"step_{optimizer_step}.pt"))

    pbar.close()
    elapsed = time.time() - start
    print(f"Finished in {elapsed/60:.2f} min")
    torch.save(_checkpoint_payload(),
               os.path.join(args.output_dir, "final_lm.pt"))

    # Persist the full run record (code version + hyperparameters + seed +
    # data version + training curve + final result) next to the checkpoint.
    # The file is the experiment log — no manual bookkeeping.
    mean_last10 = (sum(curve_losses[-10:]) / min(10, len(curve_losses))
                   if curve_losses else None)
    run_record = {
        "kind": "train_lm_autodl",
        "created_utc": created_utc,
        "elapsed_s": elapsed,
        "seed": args.seed,
        "seed_info": seed_info,
        "provenance": provenance,
        "hyperparameters": hyperparams,
        "data": data_record,
        "overfit_sanity_check": overfit_report,
        "training_curve": {
            "optimizer_step": curve_steps,
            "loss": curve_losses,
            "elapsed_s": curve_elapsed_s,
        },
        "final": {
            "optimizer_step": optimizer_step,
            "micro_step": micro_step,
            "last_loss": curve_losses[-1] if curve_losses else None,
            "mean_last10_loss": mean_last10,
            "checkpoint": os.path.join(args.output_dir, "final_lm.pt"),
            "completed": optimizer_step >= max_steps,
        },
    }
    record_path = os.path.join(args.output_dir, "run_record.json")
    write_results_json(run_record, record_path)
    print(f"[run-record] wrote {record_path}")

    if not args.skip_consistency_check:
        report = run_decode_consistency_probe(
            model, device, seq_len, args.output_dir, seed=args.seed)
        _print_consistency_report(report, title="final_lm.pt")

if __name__ == "__main__":
    main()
