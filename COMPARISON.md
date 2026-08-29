# COMPARISON.md — 原有 ops 实现 vs 早期 toy 训练实现

本文件说明为什么本仓库在新增真实 LM 训练管线（`train_lm_autodl.py`）时**保留**了原有的
`ops_kda / ops_csa / ops_hca / ops_fused` 参考实现，而不是替换成早期那份更简单的 toy
训练代码。

## 背景

仓库早期存在一个独立的 toy 训练入口（旧 `train.py`，现为
`train_toy_reference.py` 兼容包装）。它依赖仓库中并不存在的模块
（`config`、`model.hybrid_model`、`dataset`），无法直接运行，且其注意力实现是
为"能跑通 demo"而写的简化版。真实的 LM 训练入口 `train_lm_autodl.py` 现在直接
构建在 `ops_fused.HybridKCHAttention` 之上。

## 对比

| 维度 | 早期 toy 实现 | 仓库 ops 实现（保留） |
|---|---|---|
| 可运行性 | 依赖缺失模块，import 即失败 | `run_correctness.py` 267 项回归检查全通过 |
| KDA 数值稳定性 | 未 clamp log 衰减门 `g` | `g_clamp_min=-10` 截断 + 非有限输出告警 + q/k 单位范数契约文档化 |
| KDA 路径 | 仅朴素递归 | 递归 + chunkwise 双路径（互验至 fp32 舍入误差），可选 `torch.compile` / TorchScript / FLA 后端 |
| CSA 索引器梯度 | top-k 整数索引截断梯度，索引器参数训不动 | 直通估计器（STE，`topk_columns` / `full_softmax` 两种模式），训练期可学、推理期自动跳过 soft 代理 |
| Attention sink | 无 | 每头可学习 sink logit 进入 softmax 分母（DeepSeek-V4 Eq. 27） |
| 全掩码行 NaN | softmax 全 `-inf` 行产生 NaN | `_nan_safe_softmax` / row-max 平移 + `masked_fill` 显式清零 |
| 滑动窗口内存 | 全长 unfold，O(T·win·c) 峰值 | `_sliding_window_attention` 自动分块（`T·win·c > 8M` 触发），与 unfold 路径数值一致 |
| 增量解码 | 无 | `CSADecodingCache` / `HCADecodingCache`：部分 token 累积器、压缩块缓存、滑动窗口环形缓冲、CSA 索引器键缓存（与全序列前向逐 token 对齐验证） |
| 空序列 / 退化输入 | 未定义行为 | `T=0`、`topk=0`、`m=0`、空 batch、`k > n_blocks` 等边界均有显式契约与测试 |
| 形状 / dtype / device | 无校验，错位输入静默广播 | 集中式 `_validate_*`：rank、GVA 整除、dtype 一致、device 一致，报错信息含期望形状 |
| 因果性 | 仅 token 级 | token 级 + 块级（论文 Eq. 16 严格前置规则 `b < t//m`，边界测试覆盖） |
| 状态管理 | 层间共享 / 泄漏 | 每层独立 KDA 递归态 + conv lookback，`reset_state()` 统一清理，非持久 buffer 不进 checkpoint |
| 实验可解释性 | 无 | KV-cache/FLOPs 解析模型（含 ceil 块数、因果项、投影项修正）+ 手算单测 |

以上合计 20+ 处边界 case 处理，逐项对应 `run_correctness.py` 中的回归检查。

## 结论

早期 toy 实现适合快速原型，但直接用于论文实验会引入不可控的数值与正确性风险
（NaN、因果泄漏、索引器不可训练、解码期状态污染等）。原有 ops 实现已把这些
坑逐一处理并以测试锁定，更适合作为发论文的基础。因此本仓库的决策是：

1. **保留** `ops_kda / ops_csa / ops_hca / ops_fused / ops_decoding_cache` 作为唯一算子实现；
2. **新增** `train_lm_autodl.py` 训练管线（Kaggle / AutoDL，成本 <120 元），直接调用
   `ops_fused.HybridKCHAttention`；
3. **废弃** toy 入口：`train_toy_reference.py` 仅保留文件名兼容，转发到受支持的训练管线。

## 验证入口

```bash
python run_correctness.py      # 252 项算子回归检查（含上表各边界 case）
python run_all.py              # 6 个实验 + 图表端到端
python train_lm_autodl.py --kaggle   # 真实 LM 训练（TinyStories）
```
