# 实验结果影响 Bug 复查报告（追加第4轮）

- 仓库：`https://gitee.com/qwe12345678/kda-csa-hca-fusion.git`
- 日期：2026-07-13
- 本轮范围：在上一轮 3 轮复查基础上，再执行 1 轮 `pull → 检查 → 修复（如有）→ push → 再检查`。

## 第4轮流程

1. 已执行 pull：远程 master 已是最新。
2. 重新检查新增 LM 训练入口及全仓关键实验路径。
3. 发现 1 个会影响实验结果/可运行性的 bug，并已修复、提交、push。
4. 修复后再次 pull 和静态验证，未发现新的会直接影响实验结果的问题。

## 本轮发现并修复的问题

### LM 训练在 Kaggle T4 上强制 BF16 autocast

**文件**：`train_lm_autodl.py`

**问题**：脚本原先在所有 CUDA 设备上都使用：

```python
with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
```

但 Kaggle 常用 T4（sm_75）不支持 BF16。强制 BF16 可能导致运行失败、退化到慢速 emulation，或使 AutoDL/Kaggle LM 训练结果不可复现。

**修复**：新增混合精度策略：

- CUDA 且 `torch.cuda.is_bf16_supported()` 为真：使用 BF16；
- CUDA 但不支持 BF16：使用 FP16 + `GradScaler`；
- CPU：禁用 autocast，使用 FP32。

同时在 optimizer step 前对 FP16 scaler 执行 `unscale_`，再做 gradient clipping，最后 `scaler.step/update`。

**提交**：`2861d06 fix: choose safe LM autocast dtype`

**Push**：已推送到 Gitee master。

## 修复后验证

已执行：

```bash
git pull <gitee-url> master
python -m py_compile *.py
bash -n run_autodl_lm.sh
```

并执行静态断言确认 `train_lm_autodl.py` 中包含：

- `torch.cuda.is_bf16_supported()`
- `torch.float16`
- `GradScaler`
- `scaler.unscale_(optimizer)`
- `scaler.step(optimizer)`

验证结果：全部通过。

## 当前提交

```text
2861d06 fix: choose safe LM autocast dtype
060c94e docs: record 3-round experiment bug audit
66f0ea9 fix: make LM training step accounting correct
5c1916e fix: repair LM training entrypoints
a2e917c verdict: keep rigorous ops, add real LM training for AutoDL <120 CNY
```

## 结论

追加第4轮发现并修复了一个会影响 Kaggle/AutoDL LM 训练运行与结果可靠性的混合精度 bug。修复后重新 pull 与检查通过，未再发现新的会直接影响实验结果的问题。
