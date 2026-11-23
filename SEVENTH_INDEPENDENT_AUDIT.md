# 第七轮独立代码审计报告

**审计日期：** 2026-11-24  
**审计范围：** KDA/CSA/HCA 三大算子核心实现、融合架构、梯度流、数值稳定性、边界条件  
**审计方法：** 独立数学推导 + 运行时验证 + 已有测试套件回归  
**测试环境：** PyTorch 2.14.1+cpu, Python 3.12  

---

## 一、审计结论

**未发现 bug 或科研方法层面的实现错误。**

本轮审计对每个关键路径进行了独立的数学推导验证和运行时测试：

- KDA 递推公式与分块并行路径：数值差异 < 5.36e-07（fp32）
- KDA 自定义 Autograd 反向梯度：与标准 autograd 差异 < 5.07e-07（fp32）
- CSA 因果块掩码（strictly preceding）：逻辑正确
- CSA 融合投影路径 parity：差异 = 0.00e+00
- HCA 非整除序列右填充：输出正确且数值有限
- NaN 安全 softmax：全掩码行正确归零，NaN 输入正确归零
- 注意力汇聚机制（attention sink）：概率质量守恒验证通过（sum(p) + p_sink = 1.000000）
- 已有测试套件 30/30 项全部通过

---

## 二、逐项数学验证

### 2.1 KDA 递推（arXiv:2510.26692）

**论文公式：**

```
D_t     = Diag(α_t) S_{t-1}           # 逐通道衰减（Diag 与 k_t k_t^T 不可交换）
delta_t = v_t - k_t^T D_t              # delta 规则残差
S_t     = D_t + β_t k_t delta_t^T      # 状态更新
o_t     = S_t^T q_t
```

**代码（`_recurrent_kda_loop_eager`）：**

```python
S = S * g_exp_i                          # D_t = S * exp(g_t)
delta = v_i - (k_i * S).sum(-2)          # delta = v - k^T D
S = torch.addcmul(S, beta_k_i, delta)    # S = D + β k ⊗ delta
outs.append(q_i @ S)                     # o = q^T S
```

**验证：** ✅ 操作顺序完全一致。衰减先于 delta 校正（`Diag(α_t)` 与 `k_t k_t^T` 不可交换，顺序关键）。GVA 展开在衰减之后。

### 2.2 KDA 分块并行路径

**DPLR 三角求解：** `A[c,j] = Σ_k k[c,k] exp(g_c - g_j) k[j,k]`，`A ← A * β`，对角线清零，求解 `(I - A) X = A`。

**状态转移方向：**

```python
kexp_i = g_i[..., -1:, :] - g_i   # g_last - g_pos
kexp_i.exp_()                       # exp(g_last - g_pos) ∈ (0, 1]
```

由于 `g = cumsum(negative_values)` 单调递减，`g_last ≤ g_pos`，故 `g_last - g_pos ≤ 0`，`exp ≤ 1`。**衰减方向正确。** ✅

**块内输出：** `row = (q * gexp) @ S + Aqk @ (u - w @ S)` ✅

### 2.3 KDA 自定义 Autograd（反向伴随）

**9 个梯度分量逐项验证通过。** bf16 STE 校正：当 `_bf16_safe_addcmul` 的 clamp 激活时，`d(clamp(x))/dx = 0`，前向和反向使用相同的 clamped 更新，梯度正确零化。

**运行时验证：** 自定义 autograd 与标准 autograd 的梯度差异 < 5.07e-07（fp32，T=64）。

### 2.4 CSA 双分支重叠压缩（DeepSeek-V4 Eq. 11-12）

```python
Zb_prev = F.pad(Zb[:, :-1], ..., value=-inf)   # 第一块 b-branch 无前一快
out = (Sa * Ca).sum(2) + (Sb * Cb_prev).sum(2)  # 两分支组合
```

**验证：** ✅ 第一块 b-branch 为 -inf → softmax 权重 0 → 仅 a-branch 贡献。后续块正确拼接。

### 2.5 CSA 因果块掩码

`mask[t, s] = (t // m > s)` — 严格前置块。

| Token 位置 | t // m | 可见块 |
|-----------|--------|--------|
| 0..3      | 0      | 无     |
| 4..7      | 1      | {0}    |
| 8..11     | 2      | {0,1}  |
| 12..15    | 3      | {0,1,2}|

**验证：** ✅ 与论文 "each query attends to only preceding compressed KV blocks" 完全一致。

### 2.6 CSA/HCA Joint Softmax + Attention Sink（Eq. 27）

Sink 是仅在分母中的虚拟条目：`denom = Σ exp(s_i) + exp(sink)`。

**运行时验证：** 对 scores=[1.0, 2.0, -inf]、sink_logit=0.5：
- p = [0.231224, 0.628532, 0.0]
- p_sink = 0.140244
- **sum(p) + p_sink = 1.000000** ✅

概率质量守恒，-inf 条目正确归零。

### 2.7 HCA 右填充处理

当 `T` 不被 `m2` 整除时（T=10, m2=8 → pad to 16）：
- Token 0（t//8=0）：压缩分支无可视块，仅 SW 分支贡献
- Token 8（t//8=1）：压缩分支可见块 0，SW + 压缩共同贡献
- 右填充位置（10..15）的 SW 键为全零 → 右填充掩码将其标记为 -inf

**运行时验证：** Token 0 norm = 1.4142（仅 SW），Token 8 norm = 0.8996（SW + 压缩），输出全有限。✅

### 2.8 缩放因子自动选择

| 算子    | normalize_qk | qk_norm_mode | 自动 scale | 验证 |
|---------|-------------|-------------|-----------|------|
| CSA 核心 | True        | l2          | 1.0       | ✅   |
| CSA 核心 | True        | rms         | c^{-0.5}  | ✅   |
| CSA 核心 | False       | —           | c^{-0.5}  | ✅   |
| HCA 核心 | True        | l2          | 1.0       | ✅   |
| HCA 核心 | True        | rms         | c^{-0.5}  | ✅   |
| KDA     | —           | —           | K^{-0.5}  | ✅   |
| 索引器    | True        | —           | 1.0       | ✅   |
| 索引器    | False       | —           | c_I^{-0.5}| ✅   |

### 2.9 NaN 安全 Softmax

| 输入情况              | 输出           | 验证 |
|----------------------|---------------|------|
| 全 -inf 行           | 全 0           | ✅   |
| 全 NaN 行            | 全 0           | ✅   |
| 正常值               | 正确 softmax   | ✅   |
| 含 +inf 行           | 均匀分布       | ✅   |

### 2.10 RoPE 部分旋转（DeepSeek-V4 §2.3.3）

- 压缩 KV 条目位置：`s * m`（块起始位置）
- 查询位置：`t`
- 输出反旋转：`-t`（使贡献仅依赖相对距离 `s·m - t`）
- 旋转在归一化之后应用（与解码缓存一致）

**验证：** ✅ 论文描述一致。

---

## 三、论文方法偏差点（已记录，非 bug）

1. **CSA 索引器压缩：** 论文要求双分支重叠压缩，代码使用单分支压缩。代码注释明确记录为 "documented simplification"，索引器仅用于排名，对压缩变体不敏感。

2. **CSA 输出投影：** 论文使用分组投影，代码使用完整投影。注释说明在实验尺度下 `c*nh` 较小，完整投影等价。

---

## 四、运行时测试汇总

| 测试项目                        | 结果   |
|-------------------------------|--------|
| KDA 递推 vs 分块 parity        | 5.36e-07 |
| KDA 自定义 autograd 梯度       | 5.07e-07 |
| CSA 因果块掩码                 | PASS   |
| CSA 融合 vs 非融合投影         | 0.00e+00 |
| CSA 非整除 T 右填充           | PASS   |
| HCA 非整除 T 右填充           | PASS   |
| NaN 安全 softmax              | PASS   |
| Sink softmax 概率守恒         | 1.000000 |
| 测试套件（4 文件 30 项）       | 30/30  |

---

## 五、结论

经过对 KDA/CSA/HCA 融合架构的第七轮独立审查，**未发现任何 bug 或科研方法上的实现错误**。代码在数学正确性、数值稳定性、边界条件处理、梯度流设计等方面均表现完备，与论文方法高度一致。
