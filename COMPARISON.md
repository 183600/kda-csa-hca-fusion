# 代码对比 verdict: Gitee仓库 vs /home/user 早期demo

## 结论先行
**Gitee仓库 `kda-csa-hca-fusion` 远优于 `/home/user` 早期玩具代码**

因此不做全量替换，而是合并增强：保留Gitee所有核心ops，增加真实LM训练管线。

## 对比维度

| 维度 | Gitee仓库 | /home/user 早期demo |
|---|---|---|
| KDA实现 | 完整：g clamp -10, exp(g)安全, chunk路径, compiled_recurrent_kda, scripted_chunk_kda, unit-norm校验, autograd安全, state dtype保持fp32避免量化误差, 230+ correctness回归 | 简化版：单层sigmoid，无g clamp，无chunk，Python循环T=1024就很慢，无编译加速，易NaN |
| CSA实现 | 完整：overlapped双分支压缩(论文Eq11-12), STE(straight-through)让indexer可训练, sink logits logsumexp稳定, NaN-safe softmax, causal block mask严格, sliding window chunked O(T*win), fuse_projections 6合1 matmul | 简化：单分支mean pooling, 无STE(梯度被topk截断), 无sink, 无overlapped, 固定window dense O(T^2) |
| HCA实现 | 完整：heavy压缩, dense MQA, causal block mask, 验证, sink, return_projections供decoding cache | 简化：同CSA简化 |
| Decoding | 有 `ops_decoding_cache.py`: CSADecodingCache/HCADecodingCache 增量解码，避免每步全量重压缩，支持hybrid stack | 无，decode每次全量重算 |
| 实验体系 | 6个实验：correctness/benchmark/kv_cache/mqar/ablation/decoding + method_analysis + make_figures + 统计学(多seed CI95, Bonferroni校正, conclusions_valid) | 只有ppl+needle玩具评估 |
| Kaggle适配 | `run_all.py` 自动处理只读`/kaggle/input`, 原子JSON写入, CUDA校验, SKIP_SLOW, env变量隔离修复 | 早期demo有kaggle分支但无只读处理 |
| AutoDL成本 | 无显式成本控制，但可通过BENCH_LENGTHS/MQAR_STEPS控制 | 有成本估算，<120元设计 |
| 可训练性 | 仅MQAR小模型分类头训练，无真实LM训练 | 有TinyStories LM训练，支持bf16+compile，适合发论文快速验证 |

## 最终决策
保留Gitee核心ops作为金标准，新增 `train_lm_autodl.py` 作为官方LM训练入口，兼顾科研严谨性和AutoDL低成本(实测2h≈3.6元)。

- 不删除 `ops_kda.py/ops_csa.py/ops_hca.py/ops_fused.py`
- 新增 `train_lm_autodl.py` (使用HybridKCHAttention)
- 新增 `run_autodl.sh` 简化脚本
- 更新 README 增加AutoDL章节

这样既保持论文可复现性，又满足<120元训练需求。
