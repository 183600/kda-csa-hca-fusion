# COMPARISON：本仓库 `ops_*` 实现 vs 早期 toy 实现

> 本文档用于回答 README 中“为什么不替换原有 ops”的问题，并给出可核对的对比口径。
> 所有结论只引用本仓库当前代码、`docs/` 设计文档、测试以及已经提交的数据文件；
> 没有真实数值支撑的维度一律标注 **待实验产出**，不补造 benchmark 或质量数值。

## 0. 口径与诚实边界

- **“本仓库 `ops_*`”**：当前提交中的 `ops_kda.py`、`ops_csa.py`、`ops_hca.py`、
  `ops_fused.py`、`ops_decoding_cache.py`、`ops_rope.py`、`ops_norm.py`、
  `ops_mamba.py` 等参考实现。代码和测试是行为的事实来源。
- **“早期 toy 实现”**：本仓库历史决策记录中的 `/home/user` 早期 demo
  （见历史提交 `a2e917c` 的对比说明）。该 demo 的实现代码**不在当前仓库树中**，
  也没有与之配套的可复现实验产物。因此本文只做接口/数值契约层面的定性对比；
  任何“toy vs 本仓库”的延迟、显存、PPL、MQAR 数值对比均为 **待实验产出**。
- **`results/` 的边界**：`results/` 是 `.gitignore` 中的生成目录，当前 checkout
  中不存在。README 里引用的 `results/*.json` 都需要通过对应 runner 重新生成；
  尤其 `results/exp2_benchmark.json` 的早期版本缺少 `compute_boundary` 字段，
  必须用 `python run_benchmark.py` 重新生成后才能跨算子解读。不同
  `compute_boundary` 的耗时**不可直接比较**。
- **当前唯一已提交的数据文件**是 `results_lm/blocklen_vs_context.json`。
  它记录的是 block-length/context 的配置对照，不是 toy-vs-ops 的端到端实验。

## 1. 结论先行

保留本仓库的 `ops_kda` / `ops_csa` / `ops_hca` 作为金标准，不把核心算子替换成
早期 toy 实现；新增的 `train_lm_autodl.py` 只作为真实 LM 训练入口，复用
`ops_fused.HybridKCHAttention`。原因不是“新代码更花哨”，而是原有参考实现已经
覆盖了大量会影响正确性和可复现性的边界条件（见第 3 节 20+ 项清单），而 toy
实现的历史口径是简化版，缺少这些契约。

## 2. 分维度对比

| 维度 | 本仓库 `ops_*` | 早期 toy 实现的历史口径 | 证据 / 状态 |
|---|---|---|---|
| KDA 衰减门控 | `g` 默认 `clamp(min=-10)`；递归、chunk、compiled、scripted 路径共用契约 | 简化版 sigmoid 门控，无 `g` clamp | `ops_kda.naive_recurrent_kda` / `naive_chunk_kda`; `docs/kda-compiled-chunk.md` |
| KDA 数值精度 | fp32 计算，支持 fp64；可指定返回 state dtype，避免 fp16 状态往返误差 | 无显式计算/状态精度契约 | `ops_kda._compute_dtype`、`state_dtype` |
| KDA 分块路径 | `naive_chunk_kda`、`scripted_chunk_kda`、`compiled_chunk_kda`；修复过 `diagonal=0` 回归 | 无 chunk，纯 Python 循环 | README 2026-08 修复记录；`docs/kda-compiled-chunk.md` |
| KDA 内存策略 | `recompute_attn` 可丢弃常驻 `Aqk`，逐 chunk 重建 | 无 | `ops_kda.compiled_chunk_kda`; `test_recompute_attn.py` |
| CSA 压缩 | 两分支 overlapped 压缩（论文 Eq. 11–12），含 partial-block 与 causal padding 处理 | 单分支 mean pooling | `ops_csa.csa_compress_kv_overlapped`; `docs/csa-indexer-chunking.md` |
| CSA indexer 选择 | 确定性 top-k：同分按 block index 升序；NaN 不可选；支持 `topk=0` | 普通 `topk`，梯度被截断 | `ops_csa._canonical_topk_indices`; `docs/indexer-selector.md` |
| CSA STE | `ste_mode='topk_columns'` 默认，另有 `full_softmax`；`no_grad()` 下自动跳过软代理 | 无 STE，indexer 不可训练 | `ops_csa.naive_csa`; `docs/indexer-selector.md` |
| CSA attention sink | 共享 softmax 分母中的 log-space sink，数值稳定 | 无 sink | `ops_csa._sink_softmax`; README fairness note |
| NaN-safe softmax | 处理全 mask、`+inf`、`-inf - -inf` 等边界，返回零而不是 NaN | 普通 `torch.softmax` | `ops_csa._nan_safe_softmax`; `test_nan_safe_softmax.py` |
| 因果块掩码 | 严格前置式 `b < t // m`（论文 Eq. 16），prefill/decode/FLOPs 共用规则 | dense 或窗口关闭式掩码 | `ops_csa._causal_block_mask`; README 2026-08 修复记录 |
| 滑窗分支 | `naive_csa`/`naive_hca` 使用 unfold 联合 softmax；另有可自动分块的 `_sliding_window_attention` | 固定窗口 dense，近似 O(T²) | `ops_csa._sliding_window_scores` / `_sliding_window_attention` |
| 联合 softmax | 压缩块、滑窗 token、sink 共享一个分母（论文 Eq. 27） | 分支各自 softmax 后相加 | `ops_csa._sink_softmax` 与 `naive_csa` 的联合 softmax 拼接；README 修复记录 |
| HCA | heavy compression + dense MQA + 可选 SW + sink，支持 `return_projections` 给 decoding cache | 与早期 CSA 简化版同级的简化实现 | `ops_hca.naive_hca` |
| 部分 RoPE | `ops_rope` 对 query、压缩 KV、滑窗 key 旋转，输出逆旋转；支持增量解码 | 无 RoPE | `ops_rope.partial_rope` / `inverse_partial_rope` |
| Q/K 归一化 | L2 默认，可选 RMSNorm；含溢出回退路径 | 无显式归一化契约 | `ops_csa._qk_normalize`; `docs/rmsnorm-reference.md` |
| 增量解码 | `CSADecodingCache` / `HCADecodingCache` / `Mamba2DecodingState`；partial accumulator、压缩块缓存、滑窗 ring buffer、indexer key cache | decode 每步全量重算 | `ops_decoding_cache.py`; `docs/chunked-prefill-cache.md` |
| 分块 prefill | `forward_chunk` 支持 continuation chunk，绝对位置、联合 softmax、cache state 与逐 token 路径一致 | 无 | `docs/chunked-prefill-cache.md`; `test_chunked_prefill_cache.py` |
| Mamba-2 / 可切换记忆路由 | `memory_type='kda'`/`'mamba2'`；参考 scan、chunked SSD、官方 backend 适配 | toy 无 Mamba-2 路由 | `ops_mamba.py`; `docs/mamba2-framework.md` |
| 混合栈状态管理 | 3:1:1 布局；prefill + decode == 全序列等价；跨请求状态隔离 | 简化 hybrid，无完整 decode 状态契约 | `ops_fused.HybridAttentionStack`; `docs/decode-state-ablation.md` |
| 实验与统计 | correctness / benchmark / kv-cache / quality / ablation / decoding 等 runner；多 seed CI95、Holm、TOST、预注册 | 只有 PPL + needle 玩具评估 | README、`docs/experiment-design.md`、`docs/equivalence-testing.md` |
| 数据完整性 | near-duplicate 泄漏检测、split fingerprint、只读数据目录处理 | toy Kaggle 分支无只读处理 | `near_duplicates.py`; `docs/data-integrity.md` |
| Backend 溯源 | 显式 requested backend 失败时大声报错；`auto` 才允许回退，并记录 resolved backend | 无 | `ops_kda_backend.py`; `docs/backend-provenance.md` |

> 表中“toy 无/简化”来自历史对比记录；“本仓库 vs toy 的实测差异”栏没有可信数据，
> 因此不填数值。**延迟、显存、PPL、MQAR 的 toy-vs-ops 对比：待实验产出。**

## 3. 已处理边界 case 清单（20+）

以下清单按“边界条件 → 实现位置 → 验证材料”组织。函数名来自当前代码；文档与
测试文件均在仓库中。

1. **KDA 衰减下界 clamp**：`g = g.clamp(min=g_clamp_min)`，默认 `-10`。
   `ops_kda.naive_recurrent_kda` / `naive_chunk_kda`；`docs/kda-compiled-chunk.md`。
2. **KDA `g` 异常范围/fullgraph 条件分支**：chunk prepare 无条件 clamp，
   避免数据相关分支；`ops_kda._chunk_kda_prepare`；`test_recompute_attn.py`。
3. **KDA 递归 vs 分块一致性**：`diagonal=0` 的因果注意力矩阵，修复过回归；
   `ops_kda.naive_chunk_kda`；README 2026-08 修复记录。
4. **KDA 状态精度**：计算用 fp32/fp64，返回 state 可单独指定 dtype；
   `ops_kda._compute_dtype`、`state_dtype`。
5. **KDA 空序列 / 非法输入**：`T == 0`、shape/dtype/device 校验、
   `initial_state` 形状校验、`g_clamp_min`/`scale` 合法性校验；
   `ops_kda._validate_kda_inputs` / `_validate_kda_shapes`。
6. **KDA 加速路径回退**：TorchScript 拒绝输入时回退 eager；
   `ops_kda.scripted_chunk_kda`。
7. **CSA overlapped 压缩的 partial block**：最后一个不完整 block 用零 padding，
   并保证因果性；`ops_csa.csa_compress_kv_overlapped`。
8. **CSA 严格 causal block mask**：`b < t // m`，query 自身所在 block 不可见；
   `ops_csa._causal_block_mask`；README 2026-08 修复记录。
9. **CSA top-k 同分确定性**：同分按 block index 升序，选择与 score tensor 大小无关；
   `ops_csa._canonical_topk_indices`；`test_csa_tiebreak.py`。
10. **CSA top-k NaN 不可选**：NaN 分数填充为 `-inf` 后重跑，避免 under-fill；
    `ops_csa._canonical_topk_indices`。
11. **CSA `topk=0`**：合法的“无稀疏选择”配置，返回空索引而非崩溃；
    `ops_csa.csa_lightning_indexer`。
12. **CSA NaN-safe softmax**：全 mask 返回 0；`+inf` 用极限处理；避免
    `+inf - +inf` 与 `0 * inf`；`ops_csa._nan_safe_softmax`；
    `test_nan_safe_softmax.py`。
13. **CSA attention sink**：sink 只进分母，log-space `logaddexp` 稳定；
    `ops_csa._sink_softmax`。
14. **CSA STE 边界**：默认 `topk_columns`；`full_softmax` 可选；
    `aux_contrastive` 明确报未实现；`torch.no_grad()` 下跳过软代理；
    `ops_csa.naive_csa`。
15. **CSA 投影融合的梯度契约**：`fuse_projections=True` 时 6 个投影合 1 次
    `F.linear`；`use_ste=False` 时 `W_KV_idx`/`W_Z_idx` 的 `None` grad 语义变化
    已在 docstring 和测试中固定；`ops_csa.naive_csa`。
16. **CSA indexer query 轴分块**：`query_chunk` 只改变临时显存，不改变 top-k
    结果；`ops_csa.csa_lightning_indexer`；`docs/csa-indexer-chunking.md`。
17. **CSA/HCA 滑窗自动分块**：`T * win * c > 8M` 时 `_sliding_window_attention`
    自动按 query chunk 计算；`ops_csa._sliding_window_attention`。
18. **滑窗与压缩块的联合 softmax**：窗口 token 与压缩块共享 sink 分母；
    `ops_csa._sliding_window_scores` / `_sink_softmax`。
19. **HCA 空序列与输入校验**：`T == 0`、sink shape、RoPE 参数、几何参数校验；
    `ops_hca.naive_hca`。
20. **HCA 的 sink / NaN-safe 联合 softmax**：压缩 MQA、滑窗、sink 共用稳定分母；
    `ops_hca.naive_hca`、`ops_csa._sink_softmax`。
21. **部分 RoPE**：`rope_dim` 按 head_dim 钳位；query/压缩 KV/滑窗 key 旋转，
    输出逆旋转；`rope_base` 拒绝 NaN/Inf/非实数；`ops_rope.partial_rope` /
    `inverse_partial_rope`。
22. **Q/K 归一化溢出**：L2/RMS 归一化含幅度回退，避免 `inf/inf` 产生 NaN；
    `ops_csa._l2_normalize_overflow_fallback` / `_qk_normalize`。
23. **增量解码的 partial token**：压缩块未完成时保留 partial accumulator，
    块完成后转入 compressed-block cache；`ops_decoding_cache.CSADecodingCache` /
    `HCADecodingCache`。
24. **滑窗 ring buffer**：固定容量 `[B, win, c]`，decode 时不随上下文增长；
    `ops_decoding_cache._SlidingWindowRingBuffer`。
25. **CSA indexer key cache**：增量维护已压缩块的 indexer key，避免每步重算全前缀；
    `ops_decoding_cache.CSADecodingCache`。
26. **分块 prefill 的位置与状态**：`forward_chunk` 使用绝对位置，重建严格前置
    block mask，并保证最终 cache state 与逐 token 路径一致；
    `docs/chunked-prefill-cache.md`；`test_chunked_prefill_cache.py`。
27. **解码计算 dtype**：在 cache dtype 与 query 侧 dtype 中取最宽者并至少 fp32；
    fp64 cache 不被降精度；`ops_decoding_cache._attention_compute_dtype`。
28. **Mamba-2 scan 边界**：`T == 0`、`initial_state` 衰减、group/head 映射、
    非因果项在 `exp` 前 mask，避免 `inf * 0 = NaN`；`ops_mamba`；
    `docs/ssd-scan-boundaries.md`、`test_ssd_duality.py`。
29. **Mamba-2 chunked SSD**：log-space gating，chunk 间 associative scan；
    `ops_mamba.chunked_mamba2_scan`；`docs/mamba2-framework.md`。
30. **混合栈状态隔离**：prefill + decode == 全序列、segmented prefill 等价、
    不同 batch/请求之间状态不污染；`ops_fused.HybridAttentionStack`；
    `docs/decode-state-ablation.md`。
31. **训练/推理模式切换**：decode cache 在 `torch.no_grad()` 下走硬 top-k，
    不启用 STE 软代理；`ops_csa.naive_csa`、`ops_decoding_cache`。
32. **Dropout 未实现显式报错**：`HybridConfig.dropout != 0` 抛
    `NotImplementedError`，避免静默 no-op；`ops_fused.HybridConfig`。
33. **FP8/量化状态边界**：非有限输入拒绝、E4M3FN/E5M2 饱和/次正规处理、
    STE 恒等梯度；`kv_quant.FP8Spec`；`docs/fp8-block-scaled-state.md`。
34. **Backend 解析 provenance**：显式请求不可用 backend 时报错，不静默回退；
    `auto` 的回退结果写入 `effective_backend`；`ops_kda_backend.py`；
    `docs/backend-provenance.md`。

## 4. 现有数据盘点

### 4.1 `results_lm/blocklen_vs_context.json`

这是当前仓库中唯一直接提交的 results 类 JSON（`results/` 为生成目录）。
文件记录了两组配置在不同 context length 下的平均 block length 及 per-layer 值：

| 配置 | context L | mean_len | per-layer | fixed 参考 |
|---|---:|---:|---|---:|
| `csa_dynamic` | 2048 | 4.168441 | `L0_csa=4.515987`, `L1_csa=3.820896` | `csa_fixed (4)` = 4.0 |
| `csa_dynamic` | 4096 | 4.346581 | `L0_csa=4.736629`, `L1_csa=3.956532` | `csa_fixed (4)` = 4.0 |
| `hybrid_dynamic` | 2048 | 33.059976 | `L0_csa=4.525967`, `L1_hca=61.593985` | `hybrid_fixed (34)` = 34.0 |
| `hybrid_dynamic` | 4096 | 30.112071 | `L0_csa=4.685159`, `L1_hca=55.538983` | `hybrid_fixed (34)` = 34.0 |

说明：上表数值来自 JSON 原始字段，显示时保留 6 位小数；完整精度请读文件本身。
该文件**不包含** toy 实现数据，也不包含延迟/质量对比，因此不能用于 toy-vs-ops
的结论。**toy-vs-ops 的 block-length、延迟与质量对比：待实验产出。**

### 4.2 `results/` 下的实验产物

当前 checkout 中没有 `results/`（见 `.gitignore`）。README 中列出的
`results/exp1_correctness.json`、`results/exp2_benchmark.json` 等均为运行产物，
不应假装已提交。引用规则：

- **Exp 2**：早期 `results/exp2_benchmark.json` 缺少 `compute_boundary`，
  必须用 `python run_benchmark.py` 重新生成；不同 boundary 的行不能直接比较
  “算子延迟”。旧文件只能依赖 `make_figures.py` 的
  `_OP_BOUNDARY_FALLBACK` 做兼容展示。
- **Exp 1/3/4/5/6/8/…**：用对应 runner 重新生成后再引用；仓库中已有历史版本
  不保证与当前因果掩码、RoPE 修复后的数值一致（README 2026-08 修复记录明确
  要求重新生成 `results/`）。
- **任何缺失的 toy-vs-ops 数值、toy-vs-ops 端到端质量、当前代码与旧结果的
  逐项差异：待实验产出。**

### 4.3 `docs/` 设计文档（31 篇）

本对比使用的设计文档均在 `docs/` 下，按主题分组：

- **算子/数值**：`mamba2-framework.md`、`kda-compiled-chunk.md`、
  `dplr-transitions.md`、`ssd-scan-boundaries.md`、`csa-indexer-chunking.md`、
  `indexer-selector.md`、`rmsnorm-reference.md`、`per-channel-decay-init.md`、
  `embedding-scaling.md`、`flops-accounting.md`。
- **解码/缓存**：`chunked-prefill-cache.md`、`decoding-state-certificate.md`、
  `decode-state-ablation.md`、`kv-budget-model.md`、`fp8-block-scaled-state.md`、
  `ttft-budget.md`。
- **实验/统计**：`experiment-design.md`、`comparative-ablation.md`、
  `equivalence-testing.md`、`inference-resolution.md`、`preregistration-binding.md`、
  `seed-influence.md`、`ppl-bootstrap.md`、`ppl-calibration.md`、
  `recall-flops-audit.md`、`quantization-model.md`、`layerwise-arch-search.md`、
  `tensor-parallelism.md`。
- **工程/可复现**：`backend-provenance.md`、`data-integrity.md`、
  `reproducibility.md`。

## 5. 如何复现/补充对比数据

```bash
# 正确性（会写 results/exp1_correctness.json）
python run_correctness.py

# 延迟（重新生成带 compute_boundary 的 exp2）
python run_benchmark.py

# 其它实验
python run_kv_cache.py
python run_quality.py
ABL_MODE=ratio python run_ablation.py
python run_decoding.py

# 全部（smoke 用于快速验证管线）
RUN_ALL_PROFILE=smoke python run_all.py
```

在补齐 toy 实现的可运行代码后，若要新增 toy-vs-ops 对比，必须沿用同一
`compute_boundary`、同一 seed/budget、同一几何配置，并把结果作为新文件加入
`results/`（或专门的对比目录），不要覆盖历史结果。当前没有这些数据，相关维度
保持 **待实验产出**。
