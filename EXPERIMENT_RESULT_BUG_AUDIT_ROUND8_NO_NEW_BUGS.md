# 实验结果影响 Bug 复查报告（第 8 轮，4 轮迭代中的第 2 轮 — 未发现新 bug）

- 仓库：`https://gitee.com/qwe12345678/kda-csa-hca-fusion.git`
- 日期：2026-07-15
- 用户要求：读取每个文件，分析代码有哪些影响实验结果的 bug，如果有就修改并 push，最多重复 4 轮。
- 本轮结论：**未发现影响实验结果的新 bug，停止迭代**。

## 本轮执行流程

第 1 轮发现并修复 5 处 bug（见 `EXPERIMENT_RESULT_BUG_AUDIT_ROUND8.md`）。第 2 轮以 3 个并行子代理 + 主代理人工复核的方式，对所有源文件做了**怀疑式深度复查**——专门盯第 1 轮 "no bugs" 声称可能漏掉的细微问题：

1. **ops_kda / ops_csa / ops_hca / ops_fused 深度复查（Task 2-a）**
   - 用 fp64 数值 parity 测试交叉验证 KDA recurrent vs chunked path（max abs diff 5.5e-17，远低于 fp32 tolerance）
   - 用 bit-identity 测试验证 `compiled_recurrent_kda`、`scripted_chunk_kda` 与 naive path 完全一致（0.0 diff）
   - 手推 Neumann series `(I - N)^-1 N` 与 FLA gated-delta-rule chunk-parallel 形式等价
   - 验证 `compiled_recurrent_kda` 缓存 key 覆盖所有影响数值的字段（scale / g_clamp_min / dtype / shape / requires_grad / state_dtype / output_final_state），无 stale-cache bug
   - 验证 CSA `fuse_projections=True` vs `False` bit-identical（0.0 diff）
   - 验证 CSA STE 前向 = hard 前向（0.0 diff），后向 = d(hard) + d(soft)
   - 验证 HCA 右 pad T 到 m2 倍数时 padded block 不污染真实 token（前 original_T 个 token 输出 bit-identical）
   - 验证 hybrid 3:1:1 layout 正确（`_build_layout` 默认产生 `['kda','kda','kda','csa','hca']`）
   - 验证 `KDAHybridLayer.short_conv` 是 depthwise Conv1d（groups=d，weight shape `[d, 1, 3]`），左 pad ksize-1，causal 无未来泄漏
   - 验证 `F.normalize` 在 per-head K-dim 上应用（view BEFORE normalize），不是在 H*K 拼接维上
   - 验证 pre-norm residual：`residual = x; x = norm(x); ...; x = residual + o`（residual 是 LayerNorm 之前的输入）

2. **实验 runner 深度复查（Task 2-b）**
   - `run_quality.py` MQAR 任务：手工重构一个 batch 验证 cue 在 `seq_len-1`、target = 配对 value、chance = `1/vocab = 1/16`（uninformed 模型的正确 baseline）
   - `run_quality.py` 统计：t_crit、Bonferroni crit 与 scipy 数值匹配（`_t_crit_975(5)=2.776`、`_bonferroni_crit_q(5, 0.0125)=3.495`），one-sided 方向正确
   - `run_quality.py` softmax 步数预算：默认 `softmax_steps = steps`（公平，第 1 轮已修复的 P0-3 fix 仍然有效）
   - `run_quality.py` per-seed RNG 隔离：embed/head 在 op-specific layer 之前创建 → 各 op 在同一 seed 下 init 完全相同
   - `run_benchmark.py` op_boundary 标签与实际 timing 边界一致；CUDA events 每 iteration sync；peak = `max_memory_allocated - baseline`（baseline 在 warmup 后捕获）
   - `run_ablation.py` 7 个 ratio 都 sum 到 5；per-ratio model init 保留 embed/head 一致
   - `run_decoding.py` SoftmaxAttnDecoding prefill 应用 causal mask，decode (T_new=1) 正确跳过 mask；CUDA events（不是 per-token CPU sync）；HybridDecoding 正确 thread KDA state + per-layer CSA/HCA caches
   - `run_kv_cache.py` 解析公式与 `CSADecodingCache` / `HCADecodingCache` 实际字段 shape 匹配；FLOPs 包含输入 + 输出投影；ceil/floor 逻辑正确
   - `train_lm_autodl.py` labels = `tokens[1:]` 与 `input_ids = tokens[:-1]` 对齐；`labels[real_len-1:] = -100` 正确 mask pad；grad_accum 语义正确；AMP dtype 选择正确；所有 RNG 源 seeded
   - `run_all.py` env var snapshot/restore 在 `finally` 块里（异常时也恢复）；rc contract 正确（`None`/`0`→ok, 非零→fail, 异常→fail）；`write_json_atomic` 原子写；CWD restore 在 `finally` 块里

3. **解码 cache + 图 + 测试 深度复查（Task 2-c）**
   - `ops_decoding_cache.py` `append_step`：per-token 累积、block-boundary 压缩（CSA overlapped 2-branch via `_csa_compress_kv_overlapped_single` 匹配 `csa_compress_kv_overlapped` per-block；HCA 单分支）、`_C_comp`/`_K_IComp` 通过 `torch.cat` 更新、SW ring buffer append（L2-normalized Ca/C）、overlap_prev state（`_prev_Cb`/`_prev_Zb` detach+clone）—— 全部匹配 naive_csa/naive_hca
   - `ops_decoding_cache.py` `forward_step`：causal block mask `b < t//m` 匹配 `_causal_block_mask`；`csa_lightning_indexer` 调用参数匹配 naive_csa；sparse/dense attention einsum、sink shift（`-row_max`）、all-masked-row NaN guard、SW 单 query einsum —— 全部匹配 naive 路径
   - `ops_decoding_cache.py` `reset()`：清空全部 11 个 state 字段（6 个 acc list、`_prev_Cb`、`_prev_Zb`、`_C_comp`、`_K_IComp`、`_sw`、计数器）
   - 边界 case：topk>n_blocks（padding + valid_mask）、topk=0（早退零）、all-masked rows（all_invalid guard）、win=0（`_sw` 为 None）、n_blocks=0（sparse_out 零）、append_step T_new>1（Python loop）
   - `make_figures.py` JSON envelope+bare-array 加载正确；坐标轴 label/scale/unit 正确；统计注释正确
   - `method_analysis.py` headwise prototype residual 是标准 pre-LN；dense-CSA 简化在 `_csa_heads` docstring 中文档化；prototype 不被任何实验使用
   - `run_correctness.py` 测试 oracle 都比较正确的 tensor，tolerance 文档化（1e-4 incremental-vs-full fp32、1e-5 compressed-block fp32、1e-10 fp64 reference、1e-12 zero-grad）
   - `conftest.py` 无 state leak（`pytest_pyfunc_call` 无状态、device fixture 返回 string、`_close_figs_fixture` 关闭 figs）
   - `test_figures.py` 7 个测试 oracle 都正确
   - `run_autodl_lm.sh` 正确调用 `train_lm_autodl.py`；`--autodl` 触发 non-Kaggle profile

## 第 1 轮修复后的验证

```bash
python -m py_compile *.py           # 全部通过
python run_correctness.py            # 239/239 通过
```

针对性回归测试：

- `naive_hca(T=0, return_projections=True)` 正确 unpack 出 `(o, (C, Z))` ✓
- `naive_csa(T=0, return_projections=True)` 正确 unpack 出 6 元组 ✓
- `setup_kaggle()` 在 `SKIP_CUDA_CHECK=1` 下直接调用能正常 return ✓
- `write_results_json()` 能处理含 `torch.Tensor` 的 payload（fallback 到 `default=str`），也能处理 NaN/Inf（sanitize 为 null）✓
- `configure_torch_for_device()` 之后 `detect_env().num_threads == torch.get_num_threads()`（GPU 上也是 1）✓

## 本轮发现

**无新 bug。** 3 个子代理 + 主代理人工复核未发现任何会影响实验结果的新问题。

## 迭代终止

按用户指令"一直重复这个过程直到没有影响实验结果的 bug，最多重复 4 轮"：

- 第 1 轮：发现 5 处 bug，全部修复并 push（commit `6ab92a9`）
- 第 2 轮：未发现新 bug → **停止迭代**

总迭代轮数：2（未达到 4 轮上限）。

## 最近提交

```text
6ab92a9 fix: 5 bugs affecting experimental results (round 8, 1 of 4)
dc55627 docs: record manual no-issue audit
b4fb217 docs: record statistical fallback audit
8b5d762 fix: label statistical fallback accurately
474e3b5 docs: record no-issue audit stop
```
