# COMPARISON.md — 算子实现选择说明

本文件说明本仓库的算子实现选择：`ops_kda / ops_csa / ops_hca / ops_fused`
是唯一的算子实现，真实 LM 训练管线（`train_lm_autodl.py`）直接构建在其
之上。

## 实现入口

真实 LM 训练入口 `train_lm_autodl.py` 直接构建在
`ops_fused.HybridKCHAttention` 之上；`train_toy_reference.py` 仅保留文件名
兼容，转发到受支持的训练管线。

## 仓库 ops 实现的能力

| 维度 | 仓库 ops 实现 |
|---|---|
| 可运行性 | `run_correctness.py` 对拍检查全通过 |
| KDA 数值稳定性 | `g_clamp_min=-10` 截断 + 非有限输出告警 + q/k 单位范数契约文档化 |
| KDA 路径 | 递归 + chunkwise 双路径（互验至 fp32 舍入误差），可选 `torch.compile` / TorchScript / FLA 后端 |
| CSA 索引器梯度 | 直通估计器（STE，`topk_columns` / `full_softmax` 两种模式），训练期可学、推理期自动跳过 soft 代理 |
| Attention sink | 每头可学习 sink logit 进入 softmax 分母（DeepSeek-V4 Eq. 27） |
| 全掩码行 | `_nan_safe_softmax` / row-max 平移 + `masked_fill` 显式清零 |
| 滑动窗口 | `_sliding_window_attention` 自动分块（`T·win·c > 8M` 触发），与 unfold 路径数值一致 |
| 增量解码 | `CSADecodingCache` / `HCADecodingCache`：部分 token 累积器、压缩块缓存、滑动窗口环形缓冲、CSA 索引器键缓存（与全序列前向逐 token 对齐验证） |
| 退化输入 | `T=0`、`topk=0`、`m=0`、空 batch、`k > n_blocks` 等边界均有显式契约与测试 |
| 形状 / dtype / device | 集中式 `_validate_*`：rank、GVA 整除、dtype 一致、device 一致，报错信息含期望形状 |
| 因果性 | token 级 + 块级（论文 Eq. 16 严格前置规则 `b < t//m`，边界测试覆盖） |
| 状态管理 | 每层独立 KDA 递归态 + conv lookback，`reset_state()` 统一清理，非持久 buffer 不进 checkpoint |
| 实验可解释性 | KV-cache/FLOPs 解析模型（含 ceil 块数、因果项、投影项修正）+ 手算单测 |

## 结论

1. `ops_kda / ops_csa / ops_hca / ops_fused / ops_decoding_cache` 作为唯一算子实现；
2. `train_lm_autodl.py` 训练管线（Kaggle / AutoDL）直接调用
   `ops_fused.HybridKCHAttention`；
3. `train_toy_reference.py` 仅保留文件名兼容，转发到受支持的训练管线。

## 验证入口

```bash
python run_correctness.py      # 算子对拍检查
python run_all.py              # 全部实验 + 图表端到端
python train_lm_autodl.py --kaggle   # 真实 LM 训练（TinyStories）
```
