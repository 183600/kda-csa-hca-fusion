# Mamba-2 实验数据资产目录

本文件记录本仓库中所有 Mamba-2 相关实验数据的来源、版本、参数配置以及完整性信息。

## 1. 数据资产总览

| 资产ID | 类型 | 位置 | 负责测试 | 说明 |
|--------|------|------|----------|------|
| m2-001 | 参考实现扫描 | ops_mamba.py | test_ssd_duality.py | naive_mamba2_scan / quadratic_mamba2_scan / chunked_mamba2_scan |
| m2-002 | 参考混合器 | ops_mamba.py | checks/mamba.py | Mamba2Mixer 参考实现 |
| m2-003 | 官方后端适配器 | ops_mamba_backend.py | checks/mamba.py | OfficialMamba2Mixer 包装器 |
| m2-004 | 几何验证 | ops_mamba.py | checks/mamba.py | validate_mamba2_geometry |
| m2-005 | 后端分辨率 | ops_mamba_backend.py | checks/mamba.py | resolve_mamba_backend |

## 2. 核心实现文件

### 2.1 ops_mamba.py — 参考实现

**版本信息**：仓库主分支最新提交
**SHA256**：`868ed4142413ca964fa3f566b5dad17a9d1ac1fc3e4fd16f37a2a67421024840`
**依赖**：torch、torch.nn、torch.nn.functional

**关键函数**：

| 函数 | Purpose | 测试覆盖 |
|------|---------|----------|
| `naive_mamba2_scan` | 顺序参考扫描 | test_ssd_duality.py |
| `quadratic_mamba2_scan` | SSD 二次型_duality_oracle | test_ssd_duality.py |
| `chunked_mamba2_scan` | 分块 SSD 路径 | test_chunked_ssd.py |
| `validate_mamba2_geometry` | 几何约束验证 | checks/mamba.py |
| `Mamba2Mixer` | 参考混合器模块 | checks/mamba.py |

### 2.2 ops_mamba_backend.py — 官方后端适配器

**版本信息**：对应 mamba-ssm >= 2.2
**SHA256**：`f80496c15cc2b5784871c6ea8f956189944afdff6878063ea10e717d964d679b`
**依赖**：mamba_ssm（可选）、CUDA（官方后端）

**关键函数**：

| 函数 | Purpose | 测试覆盖 |
|------|---------|----------|
| `OfficialMamba2Mixer` | 官方 kernel 包装器 | checks/mamba.py |
| `build_official_mamba2_mixer` | 厂构函数 | checks/mamba.py |
| `transfer_reference_state` | 权重传输（部分兼容） | checks/mamba.py |
| `mamba2_official_available` | 可用性探测 | checks/mamba.py |
| `resolve_mamba_backend` | 后端解析 | checks/mamba.py |

## 3. 参数几何默认值

Mamba-2 参考配置（d_model=32）：

| 参数 | 默认值 | 计算公式 |
|------|--------|----------|
| d_state | 64 | 固定 |
| d_conv | 4 | 固定 |
| expand | 2 | d_inner = expand * d_model = 64 |
| headdim | 64 | nheads = d_inner / headdim = 1 |
| ngroups | 1 | heads_per_group = nheads / ngroups = 1 |
| dt_rank | ceil(d_model/16) = 2 | "auto" 解析 |

## 4. 后端选择协议

| 后端 | 可用条件 | 行为 |
|------|----------|------|
| `reference` | 始终可用 | CPU/CUDA 双平台 |
| `official` | CUDA + mamba-ssm | 官方 Triton 内核，CUDA 专用 |
| `auto` | - | CUDA+包 就用 official，否则 reference |

## 5. 数据完整性证书

### 5.1 扫描路径等价（test_ssd_duality.py）

- recurrent scan 与 quadratic scan：fp64 机器精度
- chunked scan 与参考路径：fp64 机器精度
- 所有路径在前向、反向、dtype 合约上完全一致

### 5.2 权重兼容性（test_mamba2_official_backend_parity.py）

- 仅限兼容：A_log, D, out_proj, dt_proj.bias / dt_bias
- 结构不兼容：in_proj（官方融合 [z,x,B,C,dt]）

## 6. 变更日志

| 版本 | 日期 | 变更 | 影响 |
|------|------|------|------|
| 2024-05 | 初始 | Mamba-2 SSD 参考实现 | ops_mamba.py |
| 2024-06 | 初始 | 官方后端适配器 | ops_mamba_backend.py |
| 2024-07 | 初始 | 完整正确性测试套件 | checks/mamba.py |

---

**备注**：
- 本清单由 Mamba-2 数据资产目录规范维护
- 任何实验数据修改需更新此文件及相应的 SHA256 校验码
- 参考论文：arXiv:2405.21060 ("Transformers are SSMs")
- 官方实现：mamba-ssm 包 (>= 2.2)

*[自动生成]*