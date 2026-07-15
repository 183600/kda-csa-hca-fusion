# 实验结果影响 Bug 复查报告（第 9 轮，4 轮迭代完成）

- 仓库：`https://gitee.com/qwe12345678/kda-csa-hca-fusion.git`
- 日期：2026-07-15
- 用户要求：读取每个文件，分析代码有哪些影响实验结果的 bug，如果有就修改并 push，一直重复直到没有影响实验结果的 bug，最多重复 4 轮。
- 最终结论：**4 轮迭代完成，共修复 3 处影响实验结果的 bug；第 4 轮未发现新 bug，停止迭代。**

## 4 轮迭代总结

| 轮次 | 范围 | 发现 | 修复 | push commit |
|------|------|------|------|-------------|
| 第 1 轮 | 全部 18 个 .py 文件（8 个并行子代理） | 1 MEDIUM | train_lm_autodl.py 权重衰减 bug | `9430a85` |
| 第 2 轮 | 测试断言正确性 + 跨文件一致性 + 数值边界 | 1 MEDIUM | run_correctness.py 容差过松 | `64cef5c` |
| 第 3 轮 | 统计公式 + FLOPs/KV-cache 公式 + 图表数据对齐 | 1 MEDIUM + 2 LOW | make_figures.py + run_kv_cache.py + run_ablation.py | `bc6d23d` |
| 第 4 轮 | conftest/test_figures + 回归验证 + 细微数值 bug 深扫 | 0（4 LOW 测试基础设施问题，不影响实验结果） | 无 | 无（停止） |

## 第 1 轮：train_lm_autodl.py 权重衰减 bug（MEDIUM）

**文件**：`train_lm_autodl.py` 第 183 行

**问题**：`optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)` 对**所有** 13.5M 参数统一施加 weight_decay=0.1，包括：
- 12.87M 参数的 tied embedding/lm_head 矩阵（占模型 95.4%）—— 由于 weight tying，embedding 即输出投影，衰减它会每步收缩 logit scale
- LayerNorm 参数（会破坏归一化稳定性）
- 偏置参数（标准做法不衰减）
- 位置偏置表 Ba/Bb/B_idx/B_pos（功能类似 embedding）

**影响**：LM 训练的 loss 曲线和 checkpoint 质量不反映模型真实潜力。`run_quality.py` 早已修复相同问题（见 `_build_param_groups` 第 56 行，注释明确说"The previous version applied weight decay uniformly to all parameters"），但修复未传播到 `train_lm_autodl.py`。

**修复**：镜像 `run_quality.py::_build_param_groups` 模式，将参数分为 decay / no_decay 两组。仅 616K 注意力/FFN 权重参数（4.6%）接收 weight_decay=0.1。验证：模块导入 OK，optimizer step OK，梯度流向两组，embedding norm 不再因 weight decay 收缩。

## 第 2 轮：run_correctness.py 容差过松（MEDIUM）

**文件**：`run_correctness.py` 第 4628、4667、4947 行

**问题**：3 个 decoding cache 正确性测试使用 `max_diff < 1e-4` 容差，但实际观察到的 max_diff 仅 ~1.2-1.8e-7：
- `test_csa_decoding_cache_correctness`: max_diff=1.79e-7, tol=1e-4
- `test_hca_decoding_cache_correctness`: max_diff=1.19e-7, tol=1e-4
- `test_hybrid_decoding_cache_matches_full_sequence`: max_diff=1.49e-7, tol=1e-4

容差比实际误差松 ~1000 倍，比文件自身 `TOL['match'][fp32]=1e-5` 标准松 10 倍。一个产生 5e-5 误差的真实 cache bug（错误 cache 更新、off-by-one causal mask、partial-accumulator bug）会静默通过。

**修复**：收紧至 `1e-5`——比 `1e-4` 紧 100 倍，仍比实际误差松 100 倍，给跨平台数值噪声留足余量。3 个测试仍全部 PASS。

## 第 3 轮：3 处修复

### 3a. make_figures.py — fig_mqar 缺少 conclusions_valid 警告（MEDIUM）

**文件**：`make_figures.py` 第 547-568 行

**问题**：`run_quality.py` 每条记录都写入 `conclusions_valid` 字段，`fig_ablation` 已在 suptitle 显示警告（"WARNING: conclusions_valid=False ... Treat as exploratory, not confirmatory."），但 `fig_mqar` 没有等价警告。README Fairness notes #3 明确说 `conclusions_valid` 是"the authoritative signal — mean_acc alone is misleading"。

**影响**：读者可能从统计功效不足的 near-chance 数据中得出强结构性结论——正是 fig_ablation 已修补防止的失败模式。

**修复**：镜像 fig_ablation 的警告逻辑，在 fig_mqar 标题中追加 `conclusions_valid=False` 警告。两个图表现在一致地警告。

### 3b. run_kv_cache.py — KDA conv lookback 硬编码（LOW）

**文件**：`run_kv_cache.py` 第 134 行

**问题**：`short_conv_state = 2 * p['d']` 硬编码 `2`，应为 `(kda_conv_ksize - 1) * p['d']`。FLOPs 路径（第 331 行）已正确参数化，但 KV-cache 路径未跟上。

**影响**：所有实验使用默认 `kda_conv_ksize=3`（此时 `2 == 3-1`），所以当前数值不变，但非默认配置的潜在 bug 已修复。

### 3c. run_ablation.py — 注释中 Bonferroni 临界值错误（LOW）

**文件**：`run_ablation.py` 第 466、471 行

**问题**：注释声称 n=3 时 Bonferroni 校正临界 t 值为 ~12.9，n=7 时为 ~4.9。实际值（经 scipy.stats.t.ppf 验证）为 ~8.3 和 ~3.4。

**修复**：更正注释数字。仅文档变更，无代码行为变化。

## 第 4 轮：未发现影响实验结果的新 bug

3 个并行子代理执行了：

1. **conftest.py + test_figures.py 审计**：发现 4 个 LOW 级测试基础设施问题（early return 跳过 slow-marking、docstring 不准确、测试只检查文件存在不检查内容、matplotlib.use 副作用），均不影响实验结果。

2. **第 1-3 轮修复回归验证**：3 处修复全部验证正确，无回归，无不良交互。Bonferroni 临界值经 scipy 验证准确。

3. **细微数值 bug 深扫**：对 ops_csa.py、ops_hca.py、ops_fused.py 逐行检查 10 类细微 bug 模式（einsum 轴错误、softmax dim 错误、mask 符号错误、权重转置错误、切片 off-by-one、reduction 轴错误、residual bug、LayerNorm 位置、scale 应用、状态更新顺序），全部验证正确。

## 审计覆盖范围

4 轮迭代共审计 18 个 Python 源文件（~18,744 行代码）：

| 文件 | 行数 | 审计轮次 | 结论 |
|------|------|----------|------|
| ops_kda.py | 1196 | R1, R4 | 无 bug |
| ops_csa.py | 1223 | R1, R4 | 无 bug |
| ops_hca.py | 312 | R1, R4 | 无 bug |
| ops_decoding_cache.py | 1284 | R1 | 无 bug（1 已记录限制） |
| ops_fused.py | 1135 | R1, R4 | 无 bug |
| run_correctness.py | 5131 | R2 | 容差修复 |
| run_benchmark.py | 606 | R1 | 1 LOW（peak-mem baseline，0.06-0.8%，未修） |
| run_decoding.py | 1244 | R1 | 无 bug |
| run_kv_cache.py | 635 | R1, R3 | conv lookback 修复 |
| run_quality.py | 1519 | R1, R3 | 无 bug（统计公式全部验证正确） |
| run_ablation.py | 769 | R1, R3 | 注释数字修复 |
| run_all.py | 449 | R1 | 无 bug |
| make_figures.py | 970 | R1, R3 | conclusions_valid 警告修复 |
| method_analysis.py | 543 | R1 | 无 bug |
| kaggle_setup.py | 815 | R1 | 无 bug |
| train_lm_autodl.py | 279 | R1 | 权重衰减修复 |
| train_toy_reference.py | 17 | R1 | 无 bug |
| conftest.py | 245 | R4 | 4 LOW（测试基础设施，不影响结果） |
| test_figures.py | 372 | R4 | 无 bug（含在 conftest 审计的 LOW 中） |

## 未修复的已知 LOW 问题（不影响实验结果）

1. **run_benchmark.py peak-mem baseline 不一致**（R1 LOW-1）：KDA 的 peak_mem_MB 包含 recurrent state，hybrid 的不包含。影响 0.06-0.8%，不影响任何实验结论。
2. **ops_decoding_cache.py SW buffer 归一化**（R1 LOW-1）：`normalize_qk=False` 时 cache 与 naive 不匹配，但所有实验均使用 `normalize_qk=True`。
3. **conftest.py 4 个测试基础设施问题**（R4 LOW-1~4）：测试选择、docstring、测试内容检查、matplotlib 副作用，均不影响实验结果。
4. **topk=0 + STE 梯度契约**（R2 LOW-1）：`topk=0` 时 indexer 参数无梯度，但无实验使用 `topk=0`。
