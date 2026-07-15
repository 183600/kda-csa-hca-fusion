# 实验结果影响 Bug 复查报告（第 8 轮，4 轮迭代中的第 1 轮）

- 仓库：`https://gitee.com/qwe12345678/kda-csa-hca-fusion.git`
- 日期：2026-07-15
- 用户要求：读取每个文件分析代码有哪些影响实验结果的 bug，如果有就修改代码并 push，一直重复这个过程直到没有影响实验结果的 bug，最多重复 4 轮。

## 本轮发现的 bug

本轮通过 7 个并行子代理 + 主代理人工复核的方式，完整审阅了 19 个 Python 源文件（ops_*, run_*, train_lm_autodl, kaggle_setup, conftest, make_figures, method_analysis, test_figures）。发现 5 处影响实验结果的 bug：

### Bug 1: `ops_hca.py` T=0 路径忽略 `return_projections` 标志

**文件**：`ops_hca.py`，行 148–149（修改前）

**问题代码**：
```python
if T == 0:
    return torch.zeros(B_, 0, nh * c, dtype=H.dtype, device=device)
```

**为什么是 bug**：函数文档明确说当 `return_projections=True` 时返回 `(output, (C, Z))` 元组。但 T=0 早退路径总是返回单个 tensor。任何调用 `o, (C, Z) = naive_hca(..., return_projections=True)` 时若 T=0，会抛出 `ValueError: not enough values to unpack`。`run_decoding.HCAAttnDecoding` 正是这样调用的，所以空 prompt prefill 或 T=0 单测会崩溃。`ops_csa.py::naive_csa` 行 695–696 有完全相同的 bug。

**实验影响**：CSA/HCA/hybrid 的 prefill 在 T=0（空 prompt）时会崩溃，丢失整次实验结果。

**修复**：T=0 路径根据 `return_projections` 分支返回对应空 tensor 元组。

### Bug 2: `ops_csa.py` T=0 路径忽略 `return_projections` 标志

**文件**：`ops_csa.py`，行 695–696（修改前）

与 Bug 1 完全相同的模式。修复方式相同（返回 6 个空 projection tensor 的元组）。

### Bug 3: `kaggle_setup.setup_kaggle()` 的 `SKIP_CUDA_CHECK` 逃生口对直接调用者失效

**文件**：`kaggle_setup.py`，行 212–228（修改前）

**问题代码**：
```python
raise RuntimeError(
    ...
    "If you intended to run on CPU, set SKIP_CUDA_CHECK=1 in the "
    "environment to bypass this guard."
)
```

**为什么是 bug**：错误消息告诉用户「设置 `SKIP_CUDA_CHECK=1` 绕过此 guard」，但 `setup_kaggle()` 本身从未读取这个环境变量。只有 `run_all._setup()`（行 110）读取它，并通过提前 return 跳过 `setup_kaggle()` 调用。**直接调用 `setup_kaggle()` 的入口**（`python kaggle_setup.py`、notebook 单元、`train_lm_autodl.py`）即使设置了环境变量也无法绕过，会在 GPU 机器上无法跑 CPU 实验。README 也把 `SKIP_CUDA_CHECK=1` 列为官方绕过方式，实现与文档不一致。

**实验影响**：在 GPU 机器上做 CPU-only 调试/对比实验时，直接调用 `setup_kaggle()` 会卡死，无法获得 CPU baseline 数据。

**修复**：在 `setup_kaggle()` 顶部检查 `SKIP_CUDA_CHECK=1`，命中则 print 并 return。

### Bug 4: `kaggle_setup.write_results_json()` 仅捕获 `ValueError`，未捕获 `TypeError`

**文件**：`kaggle_setup.py`，行 749–757（修改前）

**问题代码**：
```python
try:
    write_json_atomic(payload, target_path, indent=indent, allow_nan=False)
except ValueError as exc:
    ...
    write_json_atomic(sanitize_for_json(payload), target_path,
                      indent=indent, allow_nan=False)
```

**为什么是 bug**：`json.dumps(allow_nan=False)` 遇到 NaN/Inf 抛 `ValueError`，但遇到**不可序列化类型**（`torch.Tensor`、非 `float` 子类的 `numpy.float32` 标量、自定义对象）抛 `TypeError`。当前只捕获 `ValueError`，所以一个未 `.item()` 的 tensor 就会让整个实验结果文件写不出来，全部数据丢失。同一仓库的 `run_kv_cache._write_results` 已经正确捕获 `(TypeError, ValueError)` 并带 `default=str` 兜底——共享工具反而比局部拷贝更脆弱。

**实验影响**：任何 runner 不小心把 `torch.Tensor` 或 `numpy.float32` 直接塞进 JSON payload，就会丢掉整次实验的全部结果。

**修复**：捕获 `(TypeError, ValueError)`，先用 `sanitize_for_json`，仍失败再 fallback 到 `default=str`。

### Bug 5: `kaggle_setup.configure_torch_for_device()` 在 GPU 上不调用 `set_num_threads`，导致 provenance 元数据与实际不符

**文件**：`kaggle_setup.py`，行 325–329 + 445–463（修改前）

**问题代码**：
```python
# detect_env()
if has_gpu:
    num_threads = 1  # GPU path is single-threaded on host
else:
    num_threads = max(1, (os.cpu_count() or 1) - 1)

# configure_torch_for_device()
if device.type == "cpu":
    torch.set_num_threads(info.num_threads)   # GPU 分支没有这行
    ...
else:
    # GPU 分支只设置 cudnn.benchmark / TF32，未设置 num_threads
```

**为什么是 bug**：`detect_env()` 在 GPU 上硬编码返回 `num_threads=1`，但 `configure_torch_for_device()` 只在 CPU 分支调用 `torch.set_num_threads(info.num_threads)`。GPU 分支跳过此调用，所以 GPU 上实际 `torch.get_num_threads()` 仍是进程默认值（通常 `os.cpu_count()`）。`capture_provenance()` 写入每个实验 JSON 的 `num_threads` 字段因此与实际不符。这同时意味着 GPU 上 CPU-side ops（data loading、未命中 cuBLAS 的小 `F.linear`）会用全部 CPU 核心过饱和竞争 host，扰动延迟测量。

**实验影响**：
1. 所有实验 JSON 的 `num_threads` provenance 字段在 GPU 上是错的（报告 1，实际是 cpu_count()）——损害可复现性元数据；
2. GPU 实验的 host-side 工作过饱和，可能微扰延迟基准。

**修复**：把 `torch.set_num_threads(info.num_threads)` 提到 if/else 之外，两个分支都执行；interop threads guard 仍只 CPU 分支执行（因为该 API 只能调用一次）。

## 修复验证

```bash
python -m py_compile *.py           # 全部通过
python run_correctness.py            # 239/239 通过
```

并针对性回归了 5 个 bug 的修复：

- `naive_hca(T=0, return_projections=True)` 现在能正确 unpack 出 `(o, (C, Z))`；
- `naive_csa(T=0, return_projections=True)` 同样能正确 unpack 出 `(o, (Ca, Cb, Za, Zb, K_idx, Z_idx))`；
- `setup_kaggle()` 在 `SKIP_CUDA_CHECK=1` 下直接调用能正常 return；
- `write_results_json()` 能处理含 `torch.Tensor` 的 payload（fallback 到 `default=str`），也能处理 NaN/Inf（sanitize 为 null）；
- `configure_torch_for_device()` 之后 `detect_env().num_threads == torch.get_num_threads()`（GPU 上也是 1）。

## 提交

```
fix: 5 bugs affecting experimental results (round 8, 1 of 4)
```

- `ops_hca.py`、`ops_csa.py`：T=0 路径尊重 `return_projections`
- `kaggle_setup.py`：`setup_kaggle()` 尊重 `SKIP_CUDA_CHECK`；`write_results_json()` 捕获 `TypeError`；`configure_torch_for_device()` 在 GPU 上也调用 `set_num_threads`

## 下一轮

继续读取所有文件，复查是否仍有影响实验结果的 bug。
