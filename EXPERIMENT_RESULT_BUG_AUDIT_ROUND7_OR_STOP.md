# 实验结果影响 Bug 复查报告（追加20轮请求：发现1处后修复，再检查无问题停止）

- 仓库：`https://gitee.com/qwe12345678/kda-csa-hca-fusion.git`
- 日期：2026-07-13
- 用户要求：再来 20 轮；如果没有发现问题就停下；检查和修复交替进行；检查不仅用 shell，也要看代码。

## 执行流程

本次不是单纯 shell 空跑，而是先人工阅读关键代码，再结合静态检查：

1. pull 远程 master。
2. 人工阅读/复查：
   - `train_lm_autodl.py` 的数据集、loss 对齐、padding mask、optimizer-step 计数、CLI override、AMP、seed、checkpoint；
   - `run_autodl_lm.sh` 的入口脚本；
   - `run_quality.py` 的 Bonferroni / fallback 统计逻辑；
   - `run_ablation.py` 的 ablation ratios、统计显著性与日志；
   - 主要实验 runner 的 JSON 写入与错误返回。
3. shell/静态验证：
   - `python -m py_compile *.py`
   - `bash -n run_autodl_lm.sh`
   - 旧入口/旧依赖扫描
   - LM 训练关键不变量断言
   - JSON 写入路径检查
   - Bonferroni one-sided exact fallback 检查
4. 发现 1 处实验报告/日志口径问题，修复并 push。
5. 再次 pull + 检查，未发现新的会直接影响实验结果的问题，因此按“如果没有发现问题就停下”停止。

## 本轮发现并修复的问题

### Bonferroni fallback 日志仍称 `Cornish-Fisher`

**文件**：`run_quality.py`、`run_ablation.py`

**问题**：代码数值路径已改为 exact beta-CDF / bisection fallback，但日志和注释仍写：

```text
fallback=Cornish-Fisher
```

这不会改变数值计算本身，但会影响实验输出解释：当 scipy 不可用时，用户会误以为显著性临界值来自 Cornish-Fisher 近似，而实际是 exact beta-CDF/bisection。对于统计结果的可审计性和复现实验说明，这是错误的实验报告元数据。

**修复**：

- 将注释改为 `exact beta-CDF/bisection fallback`；
- 将日志改为：

```text
fallback=exact-beta-bisection
```

**提交**：`8b5d762 fix: label statistical fallback accurately`

**Push**：已推送到 Gitee master。

## 修复后验证

修复后再次执行：

```bash
git pull <gitee-url> master
python -m py_compile *.py
bash -n run_autodl_lm.sh
```

并静态断言：

- `run_quality.py` / `run_ablation.py` 中存在 `fallback=exact-beta-bisection`；
- 不再存在 `fallback=Cornish-Fisher`；
- Bonferroni helper 保持 one-sided：`target_p = 1.0 - alpha`；
- fallback exact path 保留 `_student_t_cdf` / `_betai`；
- LM 训练入口的 loss/padding/seed/AMP/optimizer-step 检查仍通过；
- 生产 runner 未直接 `json.dump` 写结果。

验证结果：全部通过。

## 最近提交

```text
8b5d762 fix: label statistical fallback accurately
474e3b5 docs: record no-issue audit stop
2042776 docs: record additional 20-round audit
4c9bec9 docs: record round 6 experiment bug audit
f2dbb83 fix: seed LM training reproducibly
```

## 结论

本次追加流程中发现并修复了 1 处会影响实验统计输出解释的日志/注释口径问题；修复后重新检查未发现新的会直接影响实验结果的问题，因此停止，没有继续空跑剩余轮次。
