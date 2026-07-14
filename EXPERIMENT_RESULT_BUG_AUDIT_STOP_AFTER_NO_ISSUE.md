# 实验结果影响 Bug 复查报告（追加20轮请求：第1轮无问题即停止）

- 仓库：`https://gitee.com/qwe12345678/kda-csa-hca-fusion.git`
- 日期：2026-07-13
- 用户要求：再来 20 轮；如果没有发现问题就停下；检查和修复交替进行；检查不仅用 shell，也要看代码。

## 执行结论

本次从远程 master pull 后，执行了第 1 轮人工代码审查 + shell/静态验证。未发现新的会直接影响实验结果的 bug，因此按照“如果没有发现问题就停下”的要求停止，没有继续空跑后续 19 轮。

## 本轮人工代码审查内容

人工阅读并重点检查了以下文件/逻辑：

1. `train_lm_autodl.py`
   - 检查 next-token LM 目标是否仍正确对齐：`input_ids=tokens[:-1]`，`labels=tokens[1:]`，loss 使用全位置 `logits.reshape(...)` 对 `labels.reshape(...)`。
   - 检查 padding target 是否置为 `-100`，避免 GPT-2 `pad_token=eos_token` 时把 padding/eos 当作真实训练目标。
   - 检查 `max_steps` 是否按 optimizer steps 计数，并确认每个 optimizer step 内部执行 `grad_accum` 个 micro-batch。
   - 检查 Kaggle/AutoDL CLI overrides 是否统一生效。
   - 检查 BF16/FP16 autocast 策略：支持 BF16 才使用 BF16，否则 CUDA 使用 FP16 + GradScaler，CPU 禁用 autocast。
   - 检查 seed 是否固定，并确认 DataLoader shuffle 使用独立 generator。

2. `run_autodl_lm.sh`
   - 检查脚本不再调用不存在的 `train.py` / `evaluate.py`。
   - 检查脚本调用正式入口 `train_lm_autodl.py`。
   - 检查 shell 语法。

3. `run_quality.py` / `run_ablation.py`
   - 检查 Bonferroni 仍为 one-sided above-chance 口径。
   - 检查 fallback 仍保留 dependency-free Student-t CDF / inverse-CDF 路径。
   - 检查结果写入仍为严格/原子 JSON 写入。

4. 生产实验 runner：`run_benchmark.py`、`run_quality.py`、`run_ablation.py`、`run_decoding.py`、`run_kv_cache.py`、`run_all.py`
   - 检查没有直接 `json.dump` 写生产结果路径。
   - 检查主要实验 runner 在错误/不完整结果时能返回非零。

## Shell / 静态验证

已执行并通过：

```bash
git pull <gitee-url> master
python -m py_compile *.py
bash -n run_autodl_lm.sh
```

并执行静态断言，确认：

- 不存在旧的可执行入口命令 `python train.py` / `evaluate.py --ckpt`；
- 不存在旧 toy 缺失模块导入 `from config import`、`from model.hybrid_model import`、`from dataset import`；
- `train_lm_autodl.py` 保留：
  - `--seed`
  - `random.seed(args.seed)`
  - `torch.manual_seed(args.seed)`
  - `loader_gen.manual_seed(args.seed + 1_000_000)`
  - 正确 next-token loss 对齐
  - padding label mask
  - optimizer-step 计数
  - safe autocast / GradScaler
- 生产实验 runner 不直接使用 `json.dump` 写结果。

## pytest 限制

当前沙箱仍无法完整运行 pytest，因为环境没有安装 `torch`，且 Python 为 3.13（项目要求 `<3.13`）。建议在 Python 3.10–3.12 且安装依赖后运行：

```bash
pip install -e .[dev]
pytest -q -m "not slow"
python run_correctness.py
python test_figures.py
```

## 最终结论

本轮人工代码审查和静态/shell 验证均未发现新的会直接影响实验结果的问题。按照用户“如果没有发现问题就停下”的要求，本次在第 1 轮后停止。
