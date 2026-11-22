# Laya Kaggle 微调：代码与运行路线

本页按当前仓库实现描述 GPU 微调路径。实时 kernel 状态和额度会变化；每次启动前都必须查询 Kaggle CLI，不要把文档里的历史额度或运行状态当作实时值。原始 capture、准备数据、checkpoint、预测和完整运行报告保留在 Kaggle。

## 当前进度（2026-11-23）

- 已完成 2 个配对轮次，共 4 次真实 CUDA 微调：v50 与 v51 的 RLCD、soft-ce 分支均使用 Tesla T4，并完成冻结评估。模型 checkpoint、输入和运行产物保留在 Kaggle。
- v50：每支 753 训练行、203 留出行；实际 GPU 时间 0.032350h 和 0.029335h。两支准确率 15.27%、balanced accuracy 33.33%、macro-F1 0.0883，203 笔费用后净 PnL 均为 -6394.925 bps；预测全为 short。校准后 Brier 由 0.7593 降至约 0.6833，ECE 由约 0.312 降至约 0.212，交易效果仍不合格。
- v51：准备出 132 条训练行（实际模型训练 119 条，另有 13 条用于校准）和 10 条留出行；新增 Relay 训练样本仅 1 条，训练/评估窗口分别有 845/194 条与累计记录重叠。两支各训练约 0.009073h/0.005686h；准确率 10%、balanced accuracy 33.33%、macro-F1 0.0606，10 笔费用后净 PnL 均为 -253.985 bps。该留出集样本很少，不作为充分效果结论。
- v50 结果已发布到累计 paper-trade dataset version 52：新增 131 条决策记录、累计 1,153 条（1,138 有效、15 无效），manifest SHA-256 `77e0cb652577437c13b363caa5bc31743434371851c90e0a22957de6b18ee3c1`。v51 结果已发布到 version 53：新增 10 条，累计 1,163 条（1,148 有效、15 无效），manifest SHA-256 `513f5ccad121cc1a122c9d72cd6af893427e0d268035912d02867d78e1e35a2c`。
- feedback v2 的失败原因已处理：历史累计集有重复时间戳（例如 `1795198560000`），发布器按时间戳分组、按完整 source record 匹配，不再假定时间戳唯一。递归结果反馈由轮次名识别，不依赖 GPU notebook slug。
- v51 递归构建器曾因行级 source hash 条件排除 1,007 条累计行，直接缩小了训练样本。现已删除该行级 hash 过滤；这类 SHA 只保留为运行报告中的来源记录。
- Kaggle Relay 同步 kernel v109 已完成，发布 Relay increment version 52：新增 46 个 capture ID，累计 481 个归档；manifest SHA-256 为 `e5371f09ea520b51fe3191fddc2e41988ace8a0233980b9aaa46f2c87cee4466`。原始行情只留 Kaggle。
- 递归轮次 v52 的 notebook version 5 因 Relay manifest SHA 不匹配而在训练前失败；version 6 排队后被取消（`CANCEL_ACKNOWLEDGED`），都没有开始微调或消耗本轮 GPU 额度。已移除生成器中对累计集、Relay 清单与行级哈希的硬性相等校验，时间戳冲突样本改为跳过，不再因单行冲突终止整轮。
- 修正版已更新到既有 GPU notebook `minglinl99/laya-gpu-loss-ablation-capture-v50` 的 version 7，保持 T4 GPU；当前 `QUEUED`、尚无日志，尚未训练。Experience 输入为 version 53，Relay 输入为 version 52；清单 SHA 仅记录实际值，不作为是否训练的门槛。RLCD 与 soft-ce 各计划 1.5 GPU 小时。
- Kaggle 官方 GPU 额度（2026-11-23）：已用 2.69h，剩余 27.31/30h，刷新时间 2026-11-28 00:00 UTC。额度充足，GPU 优先。
- 后续每一轮微调都更新这一个既有 GPU notebook 的版本，不新建按轮次命名的 GPU notebook；CPU 只用于必要的数据同步与发布，不用于模型微调。当前下一步是监控 version 7；获得 CUDA 后确认两支是否进入训练，并记录训练量与冻结评估结果。
## 当前实现入口

| 文件 | 用途 |
| --- | --- |
| `scripts/build_laya_kaggle_gpu_loss_ablation.py` | 生成私有 Kaggle T4 notebook 和 metadata；本轮比较 RLCD 与 soft-ce |
| `scripts/laya_market_finetune.py` | 在 Kaggle 校验输入、准备/读取切分、微调 Laya 并评估留出数据 |
| `scripts/build_laya_kaggle_model_feedback.py` | 冻结评估后在 Kaggle 内核验逐样本预测与成熟真实 outcome，并发布累计数据集新版 |
| `scripts/build_laya_kaggle_recursive_iteration.py` | 将累计真实标签与新 Relay 时间段组合，再启动下一轮双分支 GPU 微调 |
| `bulk_engine/src/laya_prepare.rs` | 从 Relay capture 生成时间排序的训练集与评估集 |
| `scripts/build_laya_kaggle_paper_trade_dataset.py` | 从固定的历史 v24 运行构建 v31 paper-trade 回放数据；它不是当前 v50 模型结果的自动追加器，也不负责训练模型 |

## v50 配对实验（已完成，作为迭代基准）

生成器的目标为 `minglinl99/laya-gpu-loss-ablation-capture-v50`。它创建私有 notebook，启用 GPU 与 Internet，设置 `NvidiaTeslaT4`，并挂载两个冻结的 Relay capture datasets。生成器把当前仓库里的 Rust 源码和微调脚本嵌入 notebook，因此 notebook 运行时不依赖本机工作区。

该轮固定 15 分钟决策间隔、15 分钟标签期限、edge 0 bps、训练比例 80%、encoder 冻结、micro-batch 16、gradient accumulation 8、1 epoch、seed 20261122；只比较 `rlcd` 和 `soft-ce`。每支默认计划上限 1.5 GPU 小时，配对总计划 3 小时。Notebook 记录实际 Relay manifest SHA、训练和评估行数及标签计数；两支复用同一数据切分。

## 生成、提交和监控

先从官方 CLI 读取新鲜额度，并核对其他运行或排队的 GPU kernel：

```sh
uvx --from kaggle --with six kaggle quota
uvx --from kaggle --with six kaggle kernels status OWNER/KERNEL-SLUG
```

依据本次查询填写已用和其他预留小时。以下 SHA 与参数仅对应冻结的 v50 snapshot；开始新一轮时必须先核实目标 manifest 与额度，不可照抄过期数字。

```sh
python3 offline_rl/scripts/build_laya_kaggle_gpu_loss_ablation.py \
  --output-dir /tmp/kaggle-kernels \
  --quota-used-hours FRESH_USED_HOURS_FROM_QUOTA \
  --reserved-gpu-hours VERIFIED_OTHER_GPU_RESERVATIONS

uvx --from kaggle --with six kaggle kernels push \
  -p /tmp/kaggle-kernels/laya-gpu-loss-ablation-capture-v50
uvx --from kaggle --with six kaggle kernels status \
  minglinl99/laya-gpu-loss-ablation-capture-v50
uvx --from kaggle --with six kaggle kernels logs \
  minglinl99/laya-gpu-loss-ablation-capture-v50
```

不要在 kernel 仍为 `QUEUED` 时重复 push 同一个版本。只有 status 和日志证明设备实际分配后，才能报告已开始 GPU 训练；无日志、状态为 queued 或 metadata 标为 T4，都不能单独证明 CUDA 已运行。

## GPU 与 CPU 选择规则

通用微调脚本支持 `--device auto|cuda`，不提供可绕过额度门槛的 `--device cpu`。训练必须提交新鲜的 7 天额度值和计划时长；额度未知会停止运行。额度仍有剩余但计划超出可用额度时也会停止，要求缩短 GPU 计划或等待额度刷新。`auto` 在额度有余且 CUDA 可用时选择 CUDA；只有官方额度确实耗尽，或运行时明确检测到 Kaggle 未提供 CUDA，才记录证据并回退 CPU。GPU 主线 notebook 仍显式检查 CUDA，并传 `--device cuda`，避免隐式设备选择。

不要为了消耗额度重复实验。只有官方滚动 GPU 额度耗尽，或 Kaggle 明确拒绝 GPU/CUDA 分配且留有日志证据，才可为尚未完成的 GPU 兼容任务考虑 CPU 回退；此时必须先记录原因。CPU-only 的历史数据整理脚本不等于模型微调任务，也不应被当成 GPU 微调的替代项。

## 产物与验收

每个分支在 Kaggle `/kaggle/working` 下分别保留 checkpoint、`run.json` 和日志；评估预测保存为 `trading-evaluation-predictions.json`。`run.json` 持久化 `runtime_probe`（CUDA 是否可用、设备数、GPU 型号和选择设备）及额度门槛；顶层 summary 记录数据 manifest SHA、GPU 型号、配置、每支训练与评估指标、实际 GPU 小时和 `run.json` SHA。核心评估包括 accuracy、balanced accuracy、macro-F1、混淆矩阵、交易数、总费用、费用后净 PnL、单决策净 PnL、最大回撤和按时间切分的结果。运行状态或训练失败时，区分“未训练”“分支失败”和“已完成训练”，不得把准备成功当成训练成功。

训练结果冻结后才能用 held-out 标签。反馈发布器会输出新版 manifest SHA 和累计记录数，递归微调生成器要求该 SHA 与预期反馈轮次完全匹配。v50 已完成 CUDA 训练、冻结评估、反馈发布和数据集 version 52 发布；递归 GPU kernel v51 已提交，等待其日志确认数据准备与 GPU 分配。后续仍需每轮检查实际运行证据。

运行数据仅留 Kaggle。用 bounded logs 查看运行进度与摘要；不要下载 notebook outputs、原始行情、准备后的样本或 checkpoint 到本机。

## 汇报内容

每轮汇报分支数量、已开始/完成/失败数、实际 GPU 小时和剩余额度、评估指标、累计数据集版本与记录数、下一步。超过 10 分钟的运行至少每 10 分钟更新一次；训练尚未开始时明确说“未完成微调”，不填写推测效果。

## 递归迭代顺序

1. 在 Kaggle GPU 上完成当前配对微调，并保存 checkpoint、逐样本预测、真实 outcome、运行清单和冻结评估结果。
2. 记录逐样本预测和成熟真实 outcome；遇到坏行或重复时间戳就跳过并计数，不因文件哈希变化或单条记录问题中止整轮。只把真实 outcome 作为标签追加到 Kaggle 累计数据集的新私有版本。
3. 下一轮读取刚发布的数据集版本训练，并使用此前未进入训练或评估的后续时间段作为留出集；先冻结新一轮输入与切分，再开始 GPU 微调。
4. 每轮结束更新本页中的轮次、运行状态、GPU 额度、指标、数据集版本及记录数，然后提交并推送代码和文档。

若当前 kernel 仍为 `QUEUED`，继续监控同一 kernel；不得重复提交同一轮，也不得把等待当成训练完成。额度充足时继续排队/运行 GPU 主线，不启动 CPU 微调。

## v50 完成时已执行的发布顺序（留作参考）

v50 完成且两支都通过 CUDA 与冻结评估核验后，在 Kaggle 内运行反馈发布器：

```sh
python3 offline_rl/scripts/build_laya_kaggle_model_feedback.py --output-dir /tmp/kaggle-kernels
uvx --from kaggle --with six kaggle kernels push -p \
  /tmp/kaggle-kernels/laya-model-feedback-gpu-loss-ablation-capture-v50
```

从反馈 kernel 的最终 JSON 输出读取累计行数和实际 manifest SHA（仅用于记录）。以反馈轮次、最新累计数据集和 Relay 数据集生成递归配对 kernel；刷新官方 GPU 额度后，在同一个 GPU notebook 上更新版本。递归生成器跳过不可用记录和重叠时间戳，保留 15 分钟标签期限与时间隔离，再启动 RLCD/soft-ce 两支 GPU 训练。

```sh
python3 offline_rl/scripts/build_laya_kaggle_recursive_iteration.py \
  --output-dir /tmp/kaggle-kernels \
  --project-slug laya-gpu-recursive-round-v51 \
  --dataset-ref minglinl99/laya-market-paper-trade-experience \
  --expected-feedback-round gpu-loss-ablation-capture-v50 \
  --experience-manifest-sha256 PUBLISHED_EXPERIENCE_MANIFEST_SHA256 \
  --capture-increment-ref minglinl99/td-market-vps-captures-increment-20261120 \
  --relay-manifest-sha256 878d44670d2119348149d29000074bd14e38b7706816cf9025e4503ea168a59e \
  --quota-used-hours FRESH_USED_HOURS_FROM_QUOTA \
  --reserved-gpu-hours VERIFIED_OTHER_GPU_RESERVATIONS

uvx --from kaggle --with six kaggle kernels push \
  -p /tmp/kaggle-kernels/laya-gpu-recursive-round-v51
```
