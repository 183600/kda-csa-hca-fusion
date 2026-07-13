# 代码审查报告：kda-csa-hca-fusion（第三轮 — 回归修复后复审）

**仓库**：https://gitee.com/qwe12345678/kda-csa-hca-fusion
**审查日期**：2026-07-13（第三轮，基于 commit `b28bc48` 之后新增的修复）
**背景**：第二轮复审发现，第一轮的 3 个高优先级修复中，"KDA 非有限值告警"
（review-fix 1.1）本身引入了一个新的高优先级回归——它让 `compiled_recurrent_kda(...,
fullgraph=True)` 无法编译。本报告记录该回归的修复（review-fix 1.1-a）及其验证过程，
并对当前代码库状态做一次完整复审。

审查方式：在本地环境中实测复现了第二轮报告中描述的 `torch.compile` 不兼容问题
（用 A/B 对比脚本确认回归确实由 review-fix 1.1 引入而非预先存在），实施并验证修复，
补齐了此前完全没有测试覆盖的 `compiled_recurrent_kda` 的回归测试，重跑全部
230 项回归测试 + 7 项图表测试 + `pytest -m "not slow"` 全量套件，并额外发现并记录了
一个与本次修复无关的、独立存在的 pytest 兼容性问题。

---

## 一、本轮修复：review-fix 1.1-a（`torch.compile` 兼容性回归）

### 问题回顾

`ops_kda.py::_warn_if_nonfinite`（review-fix 1.1 引入）中的：

```python
if not torch.isfinite(o).all():
    _warnings.warn(...)
```

是一个依赖张量运行时数值的 Python 分支。当它被 `compiled_recurrent_kda`（`naive_recurrent_kda`
的 `torch.compile` 包装器）在 `fullgraph=True` 模式下追踪时，Dynamo 会直接报错：

```
torch._dynamo.exc.Unsupported: Data-dependent branching
```

修复前的 `naive_recurrent_kda` 返回路径不含此类分支，因此这是 review-fix 1.1
引入的真实回归，且由于 `compiled_recurrent_kda` 此前完全没有测试覆盖，未被
227 项回归测试捕获。

### 修复方案

用 `torch.compiler.is_compiling()`（PyTorch 官方文档中"在被 `torch.compile`/`torch.export`
追踪时跳过某段逻辑"的标准写法）包裹检查：

```python
def _warn_if_nonfinite(o, fn_name, stacklevel=3):
    if torch.compiler.is_compiling():
        # 被 torch.compile 追踪时，整个检查在 trace 阶段被裁剪掉，
        # 不产生任何数据依赖分支，图保持完整。
        return
    if not torch.isfinite(o).all():
        _warnings.warn(...)
```

`is_compiling()` 由 Dynamo 特殊处理（在 trace 时解析为编译期常量，而不是被当作
普通的张量谓词），所以外层 `if` 是 graph-safe 的：编译期整个分支被裁剪掉（编译后的图
里没有任何诊断开销），eager 模式下（包括直接调用 `naive_recurrent_kda`/`naive_chunk_kda`/
`scripted_chunk_kda`，以及 `compiled_recurrent_kda` 首次 trace 之前的任何 eager 调用）
仍然保留非有限值检查和告警。

### 验证结果

1. **`fullgraph=True` 编译成功**：用 T=8 的最小递归验证，`compiled_recurrent_kda(...,
   fullgraph=True)` 从抛出 `Unsupported: Data-dependent branching` 变为正常编译并运行
   （耗时 ~10-15s，属于 `torch.compile` 在 CPU 上展开 Python for 循环的正常一次性编译
   成本，与本次修复无关）。
2. **数值一致性**：编译后的输出与 `naive_recurrent_kda` 逐元素完全一致（`max|diff|=0.00e+00`），
   `output_final_state=True` 时状态也完全一致。
3. **eager 路径行为不变**：未归一化输入仍然触发告警（`naive_recurrent_kda` /
   `naive_chunk_kda` 均验证），归一化输入不产生虚假告警——review-fix 1.1 的原始诊断能力
   完全保留。

### 新增测试覆盖

补充了 `test_compiled_recurrent_kda_fullgraph`（3 项子检查：编译不报错、输出匹配、
状态匹配），这也是仓库历史上**第一条直接测试 `compiled_recurrent_kda` 的回归测试**——
第二轮复审指出的"这个函数完全没有测试覆盖"的测试盲区已经补上。由于 `torch.compile`
有真实的一次性编译成本（在本环境 CPU 上约 10-15 秒），该测试已在 `conftest.py` 的
`_SLOW_TESTS` 中标记为 `slow`，`pytest -m "not slow"` 可以跳过它以保持快速迭代循环，
`run_correctness.py`（标准入口）仍然无条件运行它。

全部 230 项回归测试（第二轮的 227 项 + 本轮新增的 3 项子检查）通过，`test_figures.py`
的 7 项测试同样全部通过。

---

## 二、复审中额外发现的问题（不在本次修复范围内，供后续跟踪）

### 2.1 【已在后续修复】`CSADecodingCache`/`HCADecodingCache` 在 pytest 下的设备比较误判

用 `pytest -m "not slow"` 跑全量测试时发现 3 个既有测试失败
（`test_csa_decoding_cache_correctness`、`test_hca_decoding_cache_correctness`、
`test_csa_decoding_cache_prefill_then_decode`），报错：

```
ValueError: CSADecodingCache.forward_step: q.device=cpu does not match
cache's device=cpu. Call cache.to(device=q.device) or move q to the cache's device.
```

根因是 `torch.device('cpu') != 'cpu'` 在当前 PyTorch 版本中返回 `True`——
`ops_decoding_cache.py` 内部某些路径存储的是裸字符串 `'cpu'`，而张量的 `.device`
属性是 `torch.device` 对象，两者用 `!=` 比较时不相等，触发了 D7 fix 里新加的
"设备一致性"防御性检查，产生假阳性。

**重要说明**：这个问题与本次审查的 3 个高优先级修复**完全无关**，用 `git stash`
回退到 review-fix 1.1-a 之前的版本（`b28bc48`）复现验证，问题依然存在——它是一个
更早期修复（D7 fix，见 `ops_decoding_cache.py` 中的注释）引入的既有缺陷。之所以
之前的复审没有发现，是因为：
  * `run_correctness.py` 的自定义 runner（`main()`）传入的 `device` 变量来自
    `configure_torch_for_device()`，实际类型恰好是字符串 `'cpu'`，与 cache 内部存储的
    字符串比较一致，不触发假阳性；
  * 只有通过 `conftest.py` 的 `device` fixture 以 pytest 方式运行时，某些测试路径
    构造出的 `torch.device` 对象与字符串比较才会触发问题。

**后续状态**：该问题已在后续实验语义修复中处理：`CSADecodingCache`、
`HCADecodingCache` 和 `_SlidingWindowRingBuffer` 在构造 / `.to(...)` 时统一把
`device` 规范化为 `torch.device(...)`，因此 pytest fixture 传入字符串 `'cpu'`
或张量 `.device` 返回 `torch.device('cpu')` 时不再误判。

### 2.2 （历史遗留维护事项，非实验结果阻断项）

第二轮报告中列出的以下维护事项不改变当前实验数值语义，后续可作为工程质量改进继续跟踪：
  * KDA 星号解包模式导致的 mypy 报错（`ops_kda.py` 中 `B, T, H, K, HV, V = *q.shape, ...`
    写法，36 处报错，根因未消除）；
  * KDA/CSA/HCA 教学版实现在 4 个文件中重复（`ops_fused.py`/`run_quality.py`/
    `run_decoding.py`/`method_analysis.py`）；
  * `run_correctness.py` 持续膨胀（本轮又新增约 70 行，目前约 4850 行）；
  * 项目仍然没有 CI 配置。

---

## 三、Lint / 测试现状汇总

| 检查项 | 结果 |
|---|---|
| `run_correctness.py`（标准入口） | 历史记录：当时 230/230 通过；当前测试数已继续增长，请以最新 `run_correctness.py` 输出为准 |
| `test_figures.py` | 历史记录：当时 7/7 通过 |
| `pytest -m "not slow" run_correctness.py` | 历史记录：当时暴露出 §2.1 的 device 比较问题；该问题已在后续修复中处理，当前状态请以最新 pytest 输出为准 |
| `ruff check ops_kda.py` | 1 处既有 E501（第二轮已记录，未变化） |
| `ruff check conftest.py` | 0 处新增问题 |
| `ruff check run_correctness.py`（新增代码范围内） | 0 处新增问题 |
| `mypy --check-untyped-defs ops_kda.py` | 报错数与第二轮持平（星号解包模式根因未变） |

---

## 四、总结

| 类别 | 状态 |
|---|---|
| review-fix 1.1-a（本轮新增） | ✅ 已修复并验证：`compiled_recurrent_kda(fullgraph=True)` 恢复可编译，eager 路径诊断能力不受影响，新增 3 项回归测试子检查填补了该函数此前完全无测试覆盖的空白 |
| review-fix 1.1 / 1.2 / 1.3（上两轮） | ✅ 保持修复状态；历史记录为当时 230 项回归测试全部通过，当前测试数已增加 |
| 新发现的独立问题（§2.1） | ⚠️ 已定位根因并记录，**不在本次修复范围内**，建议作为下一个独立任务处理 |
| 遗留中优先级维护事项（§2.2） | 非实验结果阻断项，后续可继续工程化改进 |

**核心结论**：本轮修复了第二轮复审发现的"修复 A 破坏 B"式回归，且没有引入新的同类问题——
通过为 `compiled_recurrent_kda` 补齐直接测试覆盖，弥补了导致上一次回归未被发现的
测试盲区。复审过程中额外发现的 `CSADecodingCache` 设备比较问题是一个独立、范围明确的
既有缺陷，已如实记录但刻意未在本次改动中一并修复，以保持每次修复的原子性和可审查性。

---

## 2026-07-13 后续实验语义修复记录（commit 078b770 及后续）

本节补充最近一轮针对“是否影响实验结果”的复审与修复状态，避免旧章节中的历史问题误导读者：

* `ops_decoding_cache.CSADecodingCache` / `HCADecodingCache` 的 device 存储已统一规范化为 `torch.device(...)`，此前记录的 string-vs-`torch.device` 假阳性问题已处理。
* Exp6 的 `HybridDecoding` 已为每个 CSA/HCA 子层接入 incremental decoding cache；hybrid decode 行不再是“CSA/HCA 无历史上下文”的占位结果，也不再标记为 upper bound。
* `naive_csa` 与 `CSADecodingCache.forward_step` 在 `torch.no_grad()` 推理/benchmark 下会跳过 STE soft surrogate，避免把训练代理开销计入延迟。
* Exp2 的 standalone CSA/HCA benchmark 已计入 grouped output projection，使 JSON/README 中的 `end_to_end_single_layer` 边界与代码一致。
* 主要实验路径（Hybrid、Exp4、Exp6、Exp2 CSA benchmark）已显式启用 `normalize_qk=True`，使 CSA lightning indexer 的 top-k 选择使用 cosine-style 方向相似度，而不是受 q_idx/K_idx 范数支配。
* Exp3 KV-cache `full_accounting` 已补入 incremental runtime state：partial-token accumulators 以及 CSA overlapped compression 的 previous-block state。
* `pyproject.toml` 已把 `ops_decoding_cache` 纳入 `py-modules`，保证安装态 `python -m run_decoding` 能导入缓存模块。

剩余需要解释而非静默忽略的 caveat 已写入 README / JSON metadata：Exp6 cache-enabled prefill latency 包含 correctness-first Python cache population，因此是保守 reference-wrapper 数字；Exp4/Exp5 的显著性标记主要是 vs chance baseline，不是 pairwise operator/layout superiority test。
