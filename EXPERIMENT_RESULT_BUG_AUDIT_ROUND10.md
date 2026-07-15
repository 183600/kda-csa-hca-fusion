# 实验结果影响 Bug 复查报告（第 10 轮，4 轮迭代完成）

- 仓库：`https://gitee.com/qwe12345678/kda-csa-hca-fusion.git`
- 日期：2026-07-15
- 用户要求：读取每个文件，分析代码有哪些影响实验结果的 bug，如果有就修改并 push，一直重复直到没有影响实验结果的 bug，最多重复 4 轮。
- 最终结论：**4 轮迭代完成，共修复 7 处影响实验结果的 bug；第 4 轮未发现新 bug，停止迭代。**

## 4 轮迭代总结

| 轮次 | 范围 | 发现 | 修复 | push commit |
|------|------|------|------|-------------|
| 第 1 轮 | 全部 18 个 .py 文件（9 个并行子代理）+ 运行全部实验 | 1 HIGH | run_decoding.py set_num_interop_threads 崩溃 | `befc58d` |
| 第 2 轮 | 跨文件一致性 + 运行时验证 + train_lm_autodl 深扫 + 数值边界 | 1 HIGH + 2 MEDIUM | train_lm_autodl.py 嵌入初始化 + checkpoint vocab + run_quality.py KDAAttn 缺 short_conv | `c55980d`, `3497d38` |
| 第 3 轮 | 验证 R2 修复 + make_figures/method_analysis + run_all 编排 | 2 MEDIUM + 1 LOW | fig_mqar 标题截断 + SKIP_SLOW 覆盖用户 env + env 泄漏 | `9955811` |
| 第 4 轮 | 端到端运行全部实验 + 验证全部修复 + 最后一轮深扫 | 0 | 无（停止） | 无 |

## 第 1 轮：run_decoding.py set_num_interop_threads 崩溃（HIGH）

**文件**：`run_decoding.py` 第 962 行（`bench_decoding` 函数内）

**问题**：`bench_decoding()` 在每个 (op, prefill_len) 测量单元内调用 `torch.set_num_interop_threads(1)`。该调用在两个层面失败：
1. `set_num_interop_threads` 每个进程只能调用一次
2. 必须在任何并行工作开始前调用——但 warmup 块（第 920-924 行的 `model(x_prefill)` / `model(x_new)`）已经触发了并行工作

异常被 `main()` 的 try/except 静默吞掉，导致 Exp 6 的每个 cell 都记录 error 结果，所有 latency 为 null——**Exp 6 在 CPU 上产生了零可用数据**。

**同样的 bug 已在 `run_benchmark.py` 修复**（commit `d02cebd`，"fix experimental fairness and benchmark reliability"），但修复未传播到 `run_decoding.py`。

**修复**：移除 `bench_decoding()` 内的 `set_num_interop_threads(1)` 调用及其 restore。`configure_torch_for_device()`（在 `main()` 中调用）已经通过 `_interop_threads_set` guard 在进程启动时一次性设置 inter-op threads 为 1。仅保留 `set_num_threads(1)`（intra-op，可多次调用）。验证：Exp 6 全部 20 个 cell 现在产生有效 latency 数据（之前 0/20）。

## 第 2 轮：3 处修复

### 2a. train_lm_autodl.py 嵌入初始化 bug（HIGH）

**文件**：`train_lm_autodl.py` 第 78-82 行（`LMWithHybrid.__init__`）

**问题**：`nn.Embedding` 默认 `N(0, 1)` 初始化。由于 weight tying（`lm_head.weight = embed.weight`），embedding 即输出投影，其初始化 scale 决定了 logit scale。结合 final LayerNorm（width d_model=512），默认初始化产生 std ~22.6、max ~500 的 logits。Cross-entropy loss 从 ~500 开始（**均匀基线 log(V)=10.82 的 46 倍**）并保持平坦——梯度被 25.7M 参数的 tied embedding 收缩主导，而非学习 LM 任务。

**证据**：修复前 5 步运行 loss `496.6 → 498.5`（无下降）。控制实验 V=1000：默认初始化 `127.58 → 125.75`（平坦，18× 基线）vs GPT-2 std=0.02 初始化 `6.93 → 6.91`（匹配基线）。

**修复**：添加 `nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)`（GPT-2 / nanoGPT 标准）。修复后 5 步 loss `10.92 → 10.58 → 10.30 → 10.14 → 9.96`（从均匀基线开始，稳步下降）。

### 2b. train_lm_autodl.py checkpoint 缺 vocab key（MEDIUM）

**文件**：`train_lm_autodl.py` 第 304-307 行（最终 checkpoint 保存）

**问题**：`final_lm.pt` 缺少中间 `step_{N}.pt` checkpoint 包含的 `'vocab'` key。README 将 `final_lm.pt` 标注为进一步评估的标准 artifact，因此它必须自描述。

**修复**：在最终 `torch.save` dict 中添加 `"vocab": vocab_size`，匹配中间 checkpoint 的 key 集合。

### 2c. run_quality.py KDAAttn 缺 short_conv（HIGH）

**文件**：`run_quality.py` 第 570-602 行（`KDAAttn` 类）

**问题**：`KDAAttn`（Exp 4 MQAR quality）缺少 `KDAHybridLayer`（`ops_fused.py:307`）和 `KDAAttnDecoding`（`run_decoding.py:283`）都有的因果深度wise short-conv（kernel=3, groups=d）。同样的 bug 已在 commit `2a23300`（P0-4）为 `KDAAttnDecoding` 修复，但平行修复从未应用到 `KDAAttn`。

**影响**：
1. Exp 4 KDA accuracy 在缺少 short_conv 的简化 KDA 上测量
2. Exp 4 KDA accuracy（无 conv）vs Exp 5 hybrid KDA accuracy（HybridKCHAttention 有 conv）——混淆的跨实验比较
3. `run_kv_cache.prefill_flops('kda')` 包含 short_conv FLOPs，但 Exp 4 KDA accuracy 在无 short_conv 时测量——Exp 3 FLOPs vs Exp 4 accuracy 交叉引用不一致

**修复**：在 `KDAAttn.__init__` 中添加 `self.short_conv = nn.Conv1d(d, d, kernel_size=3, padding=0, groups=d, bias=True)`，在 `forward()` 中通过因果左填充（`F.pad(x.transpose(1,2), (ksize-1, 0))`）在 q/k/v/g/beta 投影前应用。Exp 4 在完整序列上运行（非流式），因此不需要持久化 `_conv_lookback` buffer。同时在 `run_correctness.py::test_standalone_kda_gate_matches_hybrid` 中添加 short_conv 存在性断言。

## 第 3 轮：3 处修复

### 3a. make_figures.py fig_mqar 标题截断（MEDIUM）

**文件**：`make_figures.py` 第 558-568 行

**问题**：第 9 轮审计（commit `bc6d23d`）在 `fig_mqar` 标题中追加了 `conclusions_valid=False` 警告，镜像 `fig_ablation` 的 suptitle 警告。但 `fig_ablation` 宽 13 英寸（~1950 px），`fig_mqar` 仅宽 6 英寸（~900 px）。组合标题（~1294 px 宽）无法容纳——居中后左侧截断 ~186 px（20.6%），右侧截断 ~208 px（23.1%），既丢失了 "Multi-Query Associative" 前缀，也丢失了 "...not confirmatory." 后缀。

**影响**：该 bug 当前在 `figures/fig_mqar_nkv1.png` 和 `figures/fig_mqar.png` 中活跃（exp4_mqar.json 对所有 4 个 op 都有 `conclusions_valid=False`）。旨在告知读者 MQAR 数据统计功效不足的警告因截断而无法阅读——正是 R3-A 修复要防止的失败模式。

**修复**：在警告前缀前插入换行符：`validity_note = '\n' + validity_note.lstrip()`。验证：标题 bbox 从 1294 px 缩小到 788 px，在 900 px 图表内有 67/45 px 余量。

### 3b. run_all.py SKIP_SLOW 覆盖用户 env vars（MEDIUM）

**文件**：`run_all.py` 第 280, 294-295, 339 行

**问题**：当 `SKIP_SLOW=1` + CPU 时，代码**无条件**设置 `BENCH_LENGTHS='128,256,512'`、`MQAR_SEEDS='3'`、`MQAR_STEPS='100'`、`ABL_STEPS='50'`——当用户显式设置更小值时**扩展**了运行（例如 `BENCH_LENGTHS=128,256` 被扩展为 128,256,512；`MQAR_SEEDS=1` 被扩展为 3）。SKIP_SLOW 应该只**截断**（过滤掉 > 512 的长度 / cap at safe ceiling），从不**扩展**。

**影响**：用户得到与请求不同的（更大的）结果。CI 门控 `MQAR_SEEDS=10` 会静默运行 3 个 seed。

**修复**：对 seeds/steps 使用 `min(user_value, safe_ceiling)`；对 BENCH_LENGTHS 解析用户列表并过滤到 <= 512（而非硬编码）。ABL_SEEDS 保持 `max(5, user)` floor（P4 统计功效保留）。验证：用户设置的 1 seed/10 steps/128,256 不再被扩展。

### 3c. run_all.py env var 泄漏（LOW）

**文件**：`run_all.py` 第 201-212 行（mutation）vs 214-215 行（`_ensure_deps`/`_setup`）vs 237 行（try block）vs 426-440 行（finally restore）

**问题**：`run_all()` 在 try-finally 块**之前**修改 env vars（第 201-212 行）。如果 `_ensure_deps()`（可能抛 `CalledProcessError`）或 `_setup()`（可能抛 `RuntimeError`）抛出，finally 块永远不会运行，修改的 env vars 泄漏到调用者进程。

**影响**：失败的 `run_all(seeds=5, steps=200)` 调用后，进程带着设置好的 env vars。仅影响 _setup() 失败时的程序化 API 调用者。Shell 用户不受影响。

**修复**：将 env var 修改移到 try 块内（在 `_setup()` 成功后）。

## 第 4 轮：未发现影响实验结果的新 bug

端到端运行 `SKIP_CUDA_CHECK=1 SKIP_SLOW=1 MQAR_SEEDS=2 MQAR_STEPS=30 ABL_SEEDS=3 ABL_STEPS=30 BENCH_LENGTHS=128,256 python run_all.py` → **8 ok / 0 failed in 85.5s**。

### 验证结果
- **exp1_correctness.json**：241/241 PASS（239 原始 + 2 新 short_conv 断言）
- **exp2_benchmark.json**：12 entries = 6 ops × 2 lengths ✓
- **exp3_kv_cache.json**：120 entries = 4 ops × multiple T × accounting modes ✓
- **exp4_mqar.json**：4 ops × 2 seeds，含 `conclusions_valid` flag ✓
- **exp5_ablation.json**：7 ratios × 5 seeds（P4 floor 从 3 提升到 5），`conclusions_valid=True` ✓
- **exp6_decoding.json**：20 entries = 5 ops × 4 prefill_lens，**零 error/null 行** ✓
- **summary.json**：env + runs + n_ok=8 + n_fail=0 + total_time_s=85.45 ✓
- **全部 7 个 figure** 生成成功（PDF + PNG）✓
- **test_figures.py**：7/7 PASS ✓
- **fig_mqar 标题**像素级验证：现在换行到两行，行 1 x=[218,703]，行 2 x=[68,852]——均在 900 px 内 ✓

### 修复完整性交叉检查
- **set_num_interop_threads**：仅 1 处实际调用（kaggle_setup.py:483，由 `_interop_threads_set` flag + try/except 保护）。其余引用均为 "do NOT call here" 注释。
- **KDA short_conv**：影响实验的全部 3 个 KDA 模块都有（KDAHybridLayer、KDAAttn、KDAAttnDecoding）。method_analysis.py 的 HeadwiseFusedAttention 缺少但为文档化的研究原型，无实验 JSON 输出。
- **图表标题截断**：仅 fig_mqar 受影响（已修复）。其他图表使用 constrained_layout（自动处理）或短标题或足够宽（fig_ablation 13in）。
- **SKIP_SLOW 截断**：经验证（在实验开始时捕获 env vars）。用户值被保留（BENCH_LENGTHS=128,256 未扩展；MQAR_SEEDS=2 未扩展等）。仅 ABL_SEEDS 从 3→5 按 P4 统计功效 floor 提升。

## 审计覆盖范围

4 轮迭代共审计 18 个 Python 源文件（~20,377 行代码）+ 运行全部 6 个实验 + 生成全部图表：

| 文件 | 行数 | 审计轮次 | 结论 |
|------|------|----------|------|
| ops_kda.py | 1196 | R1, R2 | 无 bug |
| ops_csa.py | 1223 | R1, R2 | 无 bug |
| ops_hca.py | 312 | R1, R2 | 无 bug |
| ops_decoding_cache.py | 1284 | R1, R2 | 无 bug |
| ops_fused.py | 1135 | R1, R2 | 无 bug |
| run_correctness.py | 5147 | R1, R2, R3 | short_conv 断言添加 |
| run_benchmark.py | 606 | R1, R3 | 无 bug |
| run_decoding.py | 1244 | R1, R2, R3, R4 | set_num_interop_threads 修复 |
| run_kv_cache.py | 643 | R1, R2 | 无 bug |
| run_quality.py | 1519 | R1, R2, R3, R4 | KDAAttn short_conv 修复 |
| run_ablation.py | 769 | R1, R2 | 无 bug |
| run_all.py | 449 | R1, R3, R4 | SKIP_SLOW + env 泄漏修复 |
| make_figures.py | 986 | R1, R2, R3, R4 | fig_mqar 标题修复 |
| method_analysis.py | 543 | R1, R3 | 无 bug（原型缺 short_conv，文档化） |
| kaggle_setup.py | 815 | R1 | 无 bug |
| train_lm_autodl.py | 310 | R1, R2, R3, R4 | 嵌入初始化 + checkpoint vocab 修复 |
| train_toy_reference.py | 17 | R1 | 无 bug |
| conftest.py | 245 | R1 | 无 bug |
| test_figures.py | 372 | R1, R3, R4 | 无 bug |

## 未修复的已知 LOW 问题（不影响实验结果）

1. **run_benchmark.py peak-mem baseline 不一致**（R1 LOW-1）：KDA 的 peak_mem_MB 包含 recurrent state，hybrid 的不包含。影响 0.06-0.8%，不影响任何实验结论。
2. **ops_decoding_cache.py SW buffer 归一化**（R1 LOW-1）：`normalize_qk=False` 时 cache 与 naive 不匹配，但所有实验均使用 `normalize_qk=True`。
3. **method_analysis.py HeadwiseFusedAttention 缺 short_conv**（R3 OBS-2）：文档化的研究原型，无实验 JSON 输出。
4. **test_figures.py 缺 fig_mqar 标题截断回归测试**（R4 OBS-1）：测试覆盖缺口，不影响实验结果。
5. **summary.json env 字段为字符串 repr**（R3 Finding 3）：schema 缺陷，不影响实验结果。
