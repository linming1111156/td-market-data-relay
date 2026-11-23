# Laya Kaggle 微调：代码与运行路线

本页按当前仓库实现描述 GPU 微调路径。实时 kernel 状态和额度会变化；每次启动前都必须查询 Kaggle CLI，不要把文档里的历史额度或运行状态当作实时值。原始 capture、准备数据、checkpoint、预测和完整运行报告保留在 Kaggle。

## 当前进度（2026-11-23）

- 最新微调轮次为 `gpu-recursive-round-v54`，运行在固定 Kaggle GPU notebook `minglinl99/laya-gpu-finetune` version 13，没有新建 GPU 项目。Kaggle 日志确认实际使用 2×Tesla T4/CUDA；本轮 1,079 条训练行、10 条冻结留出行，RLCD 与 soft-ce 各完成一次微调，GPU 时间分别为 0.021055h 与 0.019288h（合计 0.040343h）。两支在 10 条留出上的 accuracy 为 0.30、balanced accuracy 为 0.333、macro-F1 为 0.154；费用后净 PnL 均为 -324.963 bps。样本很少且交易结果为负，尚无费用后盈利证据。
- 结果反馈沿用现有 notebook `minglinl99/laya-model-feedback-gpu-recursive-round-v53` version 12。Kaggle Data Explorer 显示 paper-trade version 57 Ready；v56 是先前误挂载旧轮次结果的历史版本。v57 manifest 的最新轮次为 `gpu-recursive-round-v54`，source kernel 指向固定 GPU notebook；新增 10 条，累计 1,202 条决策（1,187 有效、15 无效）。RLCD 与 soft-ce 的逐样本预测均已配对加入，且本轮模型已完成冻结评估。下一步检查现有 Relay 同步项目是否有新 capture；只有出现新的、已成熟且未用于评估的时间窗口后才启动下一轮 GPU 微调。
- 固定 Relay 同步项目 `minglinl99/laya-kaggle-relay-increment-sync-main` version 5 在 2026-11-23 03:55 UTC 完成一次增量检查，并向原累计 Relay dataset 发布 version 56：新增 14 个 capture ID，累计 543 个归档、846,341,281 输入字节、0 个无效项，remote loss records 为 0。manifest SHA-256 `d128103c6ecd8329b694da6f8ae15f928d0d3c41a3689a23908a83b1d4f1e83c`；Kaggle Data Explorer 确认 version 56，发布 session 已停止。原始 capture 仍只留 Kaggle。
- 2026-11-23 04:01 UTC 官方 Kaggle CLI 查询：GPU 已用 6.97h、剩余 23.03h/30h，额度刷新时间为 2026-11-28 00:00 UTC。已列出的四个固定项目中 GPU 微调和 Relay 同步运行均完成；反馈 Draft Session 已停止。`~/.kaggle/access_token` 已存在，官方 CLI 认证可用；凭据不进入代码、文档或提交。
- 固定 GPU notebook version 14 已使用累计 paper-trade v57 和 Relay increment v56，前序轮次为 `gpu-recursive-round-v54`。GPU 环境确认实际分配 2×Tesla T4。数据准备完成：1,089 训练行、8 条新鲜后置留出，Relay 新增 1 条可训练记录；候选评估中 219 条与累计记录重叠并已排除，累计冲突时间戳 99 条按逐行规则处理。CPU 归档准备缓存未命中，在 Kaggle `/kaggle/working` 重建，原始数据未下载到本机。
- v55 的 RLCD cell 启动后被 `KeyboardInterrupt` 中断，运行约 67 秒；未得到 `run.json`、冻结指标或可用 GPU 微调结果，不计作成功训练。soft-ce 尚未启动。此轮官方额度由已用 6.97h 增至 7.24h，当前剩余 22.76h/30h，刷新时间 2026-11-28 00:00 UTC。下一步在同一 notebook 继续完成 RLCD 与 soft-ce 分段训练；每支完成后再冻结指标和执行 feedback。若中断留下同名输出目录，先在 Kaggle session 内查看其状态并续跑/清理该分支，不能把启动回执当作完成结果。
### 历史实验记录（不代表当前运行状态）

- v50 与 v51 的 RLCD、soft-ce 分支均使用 Tesla T4 并完成冻结评估；checkpoint、输入和运行产物保留在 Kaggle。
- v50：每支 753 训练行、203 留出行；实际 GPU 时间 0.032350h 和 0.029335h。两支准确率 15.27%、balanced accuracy 33.33%、macro-F1 0.0883，203 笔费用后净 PnL 均为 -6394.925 bps；预测全为 short。校准后 Brier 由 0.7593 降至约 0.6833，ECE 由约 0.312 降至约 0.212，交易效果仍不合格。
- v51：准备出 132 条训练行（实际模型训练 119 条，另有 13 条用于校准）和 10 条留出行；新增 Relay 训练样本仅 1 条，训练/评估窗口分别有 845/194 条与累计记录重叠。两支各训练约 0.009073h/0.005686h；准确率 10%、balanced accuracy 33.33%、macro-F1 0.0606，10 笔费用后净 PnL 均为 -253.985 bps。该留出集样本很少，不作为充分效果结论。
- v50 结果已发布到累计 paper-trade dataset version 52：新增 131 条决策记录、累计 1,153 条（1,138 有效、15 无效），manifest SHA-256 `77e0cb652577437c13b363caa5bc31743434371851c90e0a22957de6b18ee3c1`。v51 结果已发布到 version 53：新增 10 条，累计 1,163 条（1,148 有效、15 无效），manifest SHA-256 `513f5ccad121cc1a122c9d72cd6af893427e0d268035912d02867d78e1e35a2c`。
- feedback v2 的失败原因已处理：历史累计集有重复时间戳（例如 `1795198560000`），发布器按时间戳分组、按完整 source record 匹配，不再假定时间戳唯一。递归结果反馈由轮次名识别，不依赖 GPU notebook slug。
- v51 递归构建器曾因行级 source hash 条件排除 1,007 条累计行，直接缩小了训练样本。现已删除该行级 hash 过滤；这类 SHA 只保留为运行报告中的来源记录。
- Kaggle Relay 同步 kernel v109 已完成，发布 Relay increment version 52：新增 46 个 capture ID，累计 481 个归档；manifest SHA-256 为 `e5371f09ea520b51fe3191fddc2e41988ace8a0233980b9aaa46f2c87cee4466`。原始行情只留 Kaggle。
- 递归轮次 v52 的 notebook version 5 因 Relay manifest SHA 不匹配而在训练前失败；version 6 排队后被取消（`CANCEL_ACKNOWLEDGED`），都没有开始微调或消耗本轮 GPU 额度。已移除生成器中对累计集、Relay 清单与行级哈希的硬性相等校验，时间戳冲突样本改为跳过，不再因单行冲突终止整轮。
- 递归轮次 v52 已在 T4 上完成 RLCD、soft-ce 两支训练和 20 条冻结留出评估：945 条训练、105 条校准；GPU 时间分别为 0.022042h、0.019862h（合计 0.041904h）。两支均预测 short，费用后净 PnL -601.234 bps，accuracy 5%、balanced accuracy 33.33%、macro-F1 0.03175；Brier 从 0.7969 降至 0.6961/0.6912，ECE 从 0.4159 降至 0.3198/0.3143。留出样本只有 20 条，交易效果仍差，校准改善不足以证明策略变好。
- v52 的成熟真实 outcome 与逐样本预测已发布到累计 paper-trade dataset version 54；Kaggle MCP 当时确认 version 54 为 Ready，notes 为 `Add verified GPU model evaluation gpu-recursive-round-v52`。Relay 输入当时为 `td-market-vps-captures` version 1 与 `td-market-vps-captures-increment-20261120` version 52。
- 2026-11-23 通过官方 Kaggle CLI 与 MCP 重新核实并更新四个私有项目：GPU 微调 `minglinl99/laya-gpu-finetune` version 6、CPU 数据准备 `minglinl99/laya-cpu-data-preparation` version 4、VPS Relay 同步 `minglinl99/laya-kaggle-relay-increment-sync-main` version 3、结果反馈 `minglinl99/laya-model-feedback-gpu-recursive-round-v53` version 3。Kaggle API 显示原项目仍存在，因此沿用固定项目并保存新版本，避免重复 notebook。全部使用 `kernels push --no-run`，新版本只保存、没有执行。
- GPU 项目固定为 `minglinl99/laya-gpu-finetune`，后续每轮继续更新同一个项目。version 5 按 cell 拆分 GPU 环境、CPU 准备数据合并、RLCD、soft-ce 和汇总；metadata 挂载累计体验、Relay captures 与 CPU 项目输出，保存时 Relay increment version 53 已发布。Draft Session 曾显示 T4 x2 On；模型训练尚未开始。
- CPU 准备项目 version 3 的 Kaggle 日志确认 Relay increment version 53 输入已处理完成：859 条训练、216 条评估行；训练标签 flat/long/short 为 658/99/102，评估标签为 154/32/30；prepared manifest SHA-256 `444957e108015018e309a6635b009b08ecefc8e6d1e65fc6101c56383bdc8078`。这是数据准备，不是模型微调。
- GPU 准备/合并 cell 在训练前发现：累计体验 version 54 的时间范围覆盖这批 212 条候选评估行；应用 15 分钟 embargo 后没有新鲜留出行，因此未开始 RLCD/soft-ce 微调。不能把旧评估行作为新留出集。首次 Draft 环境尝试因 Internet 默认关闭而无法安装依赖；在 Settings 中开启 Internet 后环境 cell 执行完成。第二个数据 cell按正确路径复用了 CPU kernel 输出，并因无新评估行退出；之后已停止 Draft Session。GPU 额度因此短时增加。
- VPS 远端同步项目 `minglinl99/laya-kaggle-relay-increment-sync-main` version 2 已完成并发布 Relay increment version 53：新增 17 个 capture ID，累计 498 个归档，805,537,813 输入字节，0 个无效项；manifest SHA-256 `26eb54039538c03bf65689dec0aeb5b2be6353fde8dc24c4aa0e87e3c0e439d0`。CPU 准备项目 version 3 正在用最新增量生成训练/评估数据。结果反馈项目 `minglinl99/laya-model-feedback-gpu-recursive-round-v53` 已保存 version 2，待新一轮 GPU 训练和冻结评估后再运行。
- GPU notebook version 6 的远端源码按独立 cells 划分 GPU 环境、数据准备、RLCD、soft-ce 与汇总，metadata 请求 T4；CPU version 4 分为环境、准备/缓存、汇总三个 Python cells；VPS 同步和结果反馈分别独立成项目。新版本只保存未执行。此前 GPU version 5 日志没有 GPU 训练分支输出，CPU version 3 的 859/216 行是数据准备结果而不是微调结果。本次查询官方额度：已用 3.71h，剩余 26.29/30h，刷新时间 2026-11-28 00:00 UTC；尚无新一轮模型微调。
- Relay 的 CPU 归档准备结果缓存于 Kaggle `/kaggle/working`，相同挂载输入与参数在同一 live session 重跑时复用；输入变化或缓存不可读时自动重建，不因缓存状态中止。所有缓存数据留在 Kaggle。
## 当前实现入口

| 文件 | 用途 |
| --- | --- |
| `scripts/build_laya_kaggle_gpu_loss_ablation.py` | 生成私有 Kaggle T4 notebook 和 metadata；本轮比较 RLCD 与 soft-ce |
| `scripts/laya_market_finetune.py` | 在 Kaggle 校验输入、准备/读取切分、微调 Laya 并评估留出数据 |
| `scripts/build_laya_kaggle_model_feedback.py` | 冻结评估后在 Kaggle 内核验逐样本预测与成熟真实 outcome，并发布累计数据集新版 |
| `scripts/build_laya_kaggle_recursive_iteration.py` | 将累计真实标签与新 Relay 时间段组合，再启动下一轮双分支 GPU 微调 |
| `scripts/build_laya_kaggle_cpu_data_preparation.py` | 在 Kaggle CPU 项目准备 Relay 时间切分数据并复用 session 缓存，供 GPU notebook 挂载 |
| `bulk_engine/src/laya_prepare.rs` | 从 Relay capture 生成时间排序的训练集与评估集 |
| `scripts/build_laya_kaggle_paper_trade_dataset.py` | 从固定的历史 v24 运行构建 v31 paper-trade 回放数据；它不是当前 v50 模型结果的自动追加器，也不负责训练模型 |

项目按职责分为 CPU 数据准备、GPU 递归微调、VPS Relay 同步、GPU 评估结果反馈。GPU 微调使用固定项目 `laya-gpu-finetune`，每轮更新其版本；notebook 按 GPU 环境、准备数据读取/缓存、RLCD、soft-ce、结果汇总拆成独立 code cells。各训练分支实时输出开始/完成事件和指标；若单支失败，可从该分支单元继续。模型微调始终使用 Kaggle GPU。

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

训练结果冻结后才能用 held-out 标签。反馈发布器会输出新版 manifest SHA 和累计记录数；递归生成器在 Kaggle 内读取并记录输入实际 SHA，不把预先抄录的 SHA 或轮次名作为启动条件。v52 已完成 CUDA 训练、冻结评估和数据集 version 54 发布。每轮仍需记录实际 CUDA 与训练结果。

运行数据仅留 Kaggle。用 bounded logs 查看运行进度与摘要；不要下载 notebook outputs、原始行情、准备后的样本或 checkpoint 到本机。

## 汇报内容

每轮汇报分支数量、已开始/完成/失败数、实际 GPU 小时和剩余额度、评估指标、累计数据集版本与记录数、下一步。超过 10 分钟的运行至少每 10 分钟更新一次；训练尚未开始时明确说“未完成微调”，不填写推测效果。

## 递归迭代顺序

1. 在 Kaggle GPU 上完成当前配对微调，并保存 checkpoint、逐样本预测、真实 outcome、运行清单和冻结评估结果。
2. 记录逐样本预测和成熟真实 outcome；遇到坏行或重复时间戳就跳过并计数，不因文件哈希变化或单条记录问题中止整轮。只把真实 outcome 作为标签追加到 Kaggle 累计数据集的新私有版本。
3. 下一轮读取刚发布的数据集版本训练，并使用此前未进入训练或评估的后续时间段作为留出集；先冻结新一轮输入与切分，再开始 GPU 微调。
4. 每轮结束更新本页中的轮次、运行状态、GPU 额度、指标、数据集版本及记录数，然后提交并推送代码和文档。

若当前 kernel 仍为 `QUEUED`，继续监控同一 kernel；不得重复提交同一轮，也不得把等待当成训练完成。额度充足时继续排队/运行 GPU 主线，不启动 CPU 微调。

## 在既有 GPU notebook 上启动下一轮

v50 完成且两支都通过 CUDA 与冻结评估核验后，在 Kaggle 内运行反馈发布器：

```sh
python3 offline_rl/scripts/build_laya_kaggle_model_feedback.py --output-dir /tmp/kaggle-kernels
uvx --from kaggle --with six kaggle kernels push -p \
  /tmp/kaggle-kernels/laya-model-feedback-gpu-loss-ablation-capture-v50
```

累计数据 version 54 和 Relay increment version 52 已确认发布。刷新官方 GPU 额度并确认没有其他运行占用后，以原 notebook slug 生成分段版本；摘要从 Kaggle 输入直接记录清单 SHA，无需先复制 SHA 到命令行。推送仍指向原 GPU notebook，不创建轮次专属 notebook。准备、RLCD、soft-ce 和汇总各自独立成 cell。

```sh
python3 offline_rl/scripts/build_laya_kaggle_recursive_iteration.py \
  --output-dir /tmp/kaggle-kernels \
  --project-slug laya-gpu-loss-ablation-capture-v50 \
  --dataset-ref minglinl99/laya-market-paper-trade-experience \
  --expected-feedback-round gpu-recursive-round-v52 \
  --capture-increment-ref minglinl99/td-market-vps-captures-increment-20261120 \
  --quota-used-hours FRESH_USED_HOURS_FROM_QUOTA \
  --reserved-gpu-hours VERIFIED_OTHER_GPU_RESERVATIONS

uvx --from kaggle --with six kaggle kernels push \
  -p /tmp/kaggle-kernels/laya-gpu-loss-ablation-capture-v50
```
