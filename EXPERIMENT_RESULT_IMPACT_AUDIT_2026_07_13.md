# 实验结果影响审查与修复报告

- 仓库：`kda-csa-hca-fusion`
- 日期：2026-07-13
- 审查范围：逐一阅读并静态分析当前仓库所有已跟踪文件（代码、配置、文档、测试）。
- 审查目标：找出会直接影响实验数值、统计结论、FLOPs/KV 口径或跨实验可比性的代码问题，并在本轮修复。

> 本环境限制：沙箱 Python 为 3.13，且未安装 `torch`；项目声明支持 Python `>=3.10,<3.13`，因此无法在本沙箱运行完整 torch 数值回归。已执行 `python -m py_compile *.py`、AST 解析，以及对 `run_kv_cache.prefill_flops` 的 dependency-free 静态导入验证。完整数值测试仍需在 Python 3.10–3.12 且安装依赖后运行。

---

## 一、本轮发现并修复的问题

### P0-1：Exp3 FLOPs 公式漏计 KDA/CSA/HCA 的 grouped output projection

**涉及文件**：`run_kv_cache.py`、`run_correctness.py`、`README.md`

**问题**：

`run_kv_cache.py::prefill_flops()` 已把 softmax baseline 的输入/输出投影计入 denominator，也把 KDA/CSA/HCA 的输入、压缩、query、attention 等主路径计入 numerator；但 KDA/CSA/HCA 都在真实模块中执行最终 grouped output projection：

- KDA：`KDAHybridLayer.o_proj` / `run_quality.KDAAttn.o` / `run_decoding.KDAAttnDecoding.o`，形状 `HV*V -> d`；
- CSA：`CSAHybridLayer.o_proj` / benchmark 的 `W_O`，形状 `csa_nh*csa_c -> d`；
- HCA：`HCAHybridLayer.o_proj` / benchmark 的 `W_O`，形状 `hca_nh*hca_c -> d`。

旧公式漏掉这些项，导致 `prefill_flops` 以及 `flops_ratio_vs_gqa_*` 对 KDA/CSA/HCA/hybrid 的 numerator 偏低。Hybrid 的漏计会按 `3*KDA + 1*CSA + 1*HCA` 叠加，直接影响 Exp3 表格与由 `make_figures.py::fig_flops()` 生成的 FLOPs 图。

**修复**：

- KDA 添加：`out_proj = 2 * T * kda_hv * kda_v * d`；
- CSA 添加：`out_proj = 2 * T * csa_nh * csa_c * d`；
- HCA 添加：`out_proj = 2 * T * hca_nh * hca_c * d`；
- Hybrid 自动继承上述修复，因为其 FLOPs 由 `prefill_flops('kda'/'csa'/'hca')` 加权求和；
- 更新 README 的 Exp3 口径说明，明确包含 input **and grouped output** projection terms；
- 更新 `test_prefill_flops_causal_block_entries` 的手算期望值；
- 新增 `test_prefill_flops_kch_output_projections` 防止回归。

**影响**：历史生成的 `results/exp3_kv_cache.json` 和 FLOPs 图需要重新生成；旧结果低估了 KDA/CSA/HCA/hybrid 的 prefill FLOPs。

---

### P0-2：Standalone Exp4/Exp6 KDA gate 参数化与 Hybrid/Exp3 不一致

**涉及文件**：`run_quality.py`、`run_decoding.py`、`run_correctness.py`

**问题**：

`ops_fused.KDAHybridLayer` 使用论文/项目文档中描述的低秩 gate 参数化：

```python
g_down: d -> K
g_up:   K -> HV*K
g = -softplus(g_up(g_down(x))) * decay_scale
```

`run_kv_cache.py::prefill_flops('kda')` 也按这个低秩参数化计 FLOPs。但 `run_quality.KDAAttn`（Exp4 standalone MQAR）和 `run_decoding.KDAAttnDecoding`（Exp6 standalone decode）实际用了直接投影：

```python
g: d -> H*K
```

这与代码注释“low-rank down/up”矛盾，也使 Exp4/Exp6 的 standalone KDA 与 hybrid KDA 层、Exp3 FLOPs 口径不是同一个 operator boundary。结果上会造成跨实验比较被 gate 参数化差异混淆：质量、延迟、参数/FLOPs 解释都不再完全对齐。

**修复**：

- `run_quality.KDAAttn`：替换为 `self.g_down` + `self.g_up`，forward 改为 `self.g_up(self.g_down(x))`；
- `run_decoding.KDAAttnDecoding`：同样替换为低秩 gate，forward 使用 `x_conv` 经过 `g_down/g_up`；
- 新增 `test_standalone_kda_gate_matches_hybrid`，检查 Exp4/Exp6 standalone KDA wrappers 不再含直接 `self.g`，且能产生有限输出。

**影响**：历史生成的 `results/exp4_mqar.json` 与 `results/exp6_decoding.json` 中 standalone KDA 行需要重新生成；旧结果使用的 KDA gate 参数化与 hybrid/Exp3 口径不一致。

---

## 二、逐文件审查摘要

| 文件 | 审查结论 | 本轮处理 |
|---|---|---|
| `.gitignore` | 仅忽略缓存、构建产物、结果目录；不会直接改变实验数值。 | 无需修改 |
| `CODE_REVIEW.md` | 历史审查记录，包含已修复问题说明；不参与运行。 | 无需修改 |
| `EXPERIMENT_IMPACT_REVIEW_20_ROUNDS.md` | 历史 20 轮复查记录；最新代码已有本轮新修复。 | 保留历史，不改写 |
| `LICENSE` | 许可证文本。 | 无需修改 |
| `README.md` | Exp3 描述未明确输出投影计入口径。 | 已更新 Exp3 FLOPs 说明 |
| `conftest.py` | pytest 设备参数、slow 标记、fixture 逻辑；未发现会改写实验结果的问题。 | 无需修改 |
| `kaggle_setup.py` | 环境检测、JSON 原子写入、sanitize、日志与 env 解析；设计能降低结果丢失/JSON 损坏风险。 | 无需修改 |
| `make_figures.py` | 图表读取 envelope/legacy JSON、空数据、错误行处理较完整；会反映输入 JSON，未发现新公式问题。 | 无需修改；需在重跑结果后重画图 |
| `method_analysis.py` | 方法解释/演示代码；KDA/CSA/HCA 公式与主实现总体一致。 | 无需修改 |
| `ops_csa.py` | CSA 主实现包含 causal mask、right padding、STE、normalize_qk、sink 稳定性、SW chunking；未发现新直接结果问题。 | 无需修改 |
| `ops_decoding_cache.py` | CSA/HCA incremental cache 已处理 partial、compressed rows、SW ring、device 规范化；未发现新直接结果问题。 | 无需修改 |
| `ops_fused.py` | Hybrid KCH 主实现；KDA low-rank gate 是本轮对齐 standalone wrappers 的基准。 | 无需修改 |
| `ops_hca.py` | HCA 主实现；causal block mask、right padding、sink/SW 逻辑与缓存路径一致。 | 无需修改 |
| `ops_kda.py` | KDA recurrent/chunk reference；已有 normalize 输入约束警告、g clamp、state dtype 处理。 | 无需修改 |
| `pyproject.toml` | 依赖版本、pytest/ruff/coverage 配置；未发现影响实验语义的问题。 | 无需修改 |
| `requirements.txt` | 依赖范围与 pyproject 基本对应。 | 无需修改 |
| `run_ablation.py` | Exp5 多 seed、统计检验、状态 reset、JSON 写入逻辑已较完整；使用 hybrid KDA low-rank gate。 | 无需修改 |
| `run_all.py` | 聚合 runner，已有 exit code 传播与结果 sanitize；未发现新直接结果问题。 | 无需修改 |
| `run_benchmark.py` | Exp2 benchmark 已在 standalone CSA/HCA 计入 output projection；未发现新直接结果问题。 | 无需修改 |
| `run_correctness.py` | 需覆盖本轮新发现的 FLOPs 输出投影与 standalone KDA gate 对齐问题。 | 已新增/更新回归测试 |
| `run_decoding.py` | Exp6 standalone KDA gate 与 Hybrid/Exp3 口径不一致。 | 已修复为 `g_down/g_up` |
| `run_kv_cache.py` | Exp3 FLOPs 漏计 KDA/CSA/HCA output projection。 | 已修复公式 |
| `run_quality.py` | Exp4 standalone KDA gate 与 Hybrid/Exp3 口径不一致。 | 已修复为 `g_down/g_up` |
| `test_figures.py` | 图表回归测试；不参与实验数值生成。 | 无需修改 |

---

## 三、已执行验证

在当前沙箱中执行：

```bash
python -m py_compile *.py
python - <<'PY'
import ast, pathlib
for p in pathlib.Path('.').glob('*.py'):
    ast.parse(p.read_text())
print('py_compile + ast parse OK')
PY
```

结果：通过。

另外用最小 `torch` stub 静态导入 `run_kv_cache`，验证：

- `prefill_flops('kda')` 相比“无 output projection”公式的差值等于 `2*T*kda_hv*kda_v*d`；
- `prefill_flops('csa')` / `prefill_flops('hca')` 总值包含正的 output projection 项；
- `prefill_flops('hybrid_kch')` 自动反映子层修复。

完整 torch 回归未能在本沙箱运行，原因：`ModuleNotFoundError: No module named 'torch'`，且 Python 版本为 3.13，不满足项目 `<3.13` 约束。

---

## 四、建议的重跑命令

在 Python 3.10–3.12 且安装依赖后执行：

```bash
pip install -e .[dev]
pytest -q -m "not slow"
python run_correctness.py
python run_kv_cache.py
python run_quality.py
python run_decoding.py
python make_figures.py
```

由于本轮修复改变了 Exp3 FLOPs、Exp4 standalone KDA、Exp6 standalone KDA 的实验语义，旧 JSON/图表不应继续作为最新结论引用。

---

## 五、最终结论

本轮逐文件审查发现 2 个会直接影响实验结果解释的问题，并已修复：

1. Exp3 FLOPs 漏计 KDA/CSA/HCA grouped output projection，导致 FLOPs ratios 低估；
2. Exp4/Exp6 standalone KDA gate 参数化与 hybrid/Exp3 不一致，导致跨实验 KDA 口径不统一。

除上述问题外，本轮未发现新的直接改变实验数值或统计结论的代码问题。完整数值回归需在项目支持的 Python + torch 环境中执行。
