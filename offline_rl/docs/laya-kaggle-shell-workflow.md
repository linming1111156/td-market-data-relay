# Laya Kaggle 递归迭代：Shell 用法

## 每轮代码是否相同

每轮的执行骨架相同：同步 Relay → 准备切分 → 在固定 GPU notebook 运行 RLCD 和 soft-ce → 冻结评估 → 将成熟真实 outcome 和逐样本预测追加到累计数据集。真正会变的是累计数据集版本、新 Relay capture、水位、上一轮 feedback ID、额度和评估结果。

Python builders 已经负责把当前仓库代码和这些参数生成 Kaggle notebook。递归训练一直更新同一个 `minglinl99/laya-gpu-finetune`；Relay 同步、CPU 数据准备和反馈也各自复用固定 Kaggle 项目。这里新增的 shell 脚本只是把重复的生成、配额读取、前置状态检查和 `kernels push` 包起来；它不在本机训练，也不下载数据。

## 首次准备

在仓库根目录运行：

```sh
chmod +x offline_rl/scripts/laya-kaggle-iteration.sh
uvx --from kaggle --with six kaggle quota
offline_rl/scripts/laya-kaggle-iteration.sh status
```

Kaggle CLI 必须使用已配置的 `~/.kaggle/access_token`。不要把 token 放进 notebook、命令行参数、日志或仓库文件。

所有 Kaggle 原始数据、训练/评估样本、checkpoint 和预测仍留在 Kaggle。Shell 只在本机 `/tmp/laya-kaggle-kernels` 生成待 push 的 notebook 和 metadata；可用 `LAYA_KAGGLE_BUILD_DIR=/另一个临时目录` 改这个目录。

## 常用命令

```sh
SCRIPT=offline_rl/scripts/laya-kaggle-iteration.sh

# 额度、固定项目状态和累计数据集文件列表
$SCRIPT status

# 查看 GPU、Relay、CPU 准备或反馈 kernel 的日志
$SCRIPT logs GPU
$SCRIPT logs RELAY
$SCRIPT logs CPU
$SCRIPT logs FEEDBACK

# 逐阶段更新固定 Kaggle 项目
$SCRIPT relay
$SCRIPT prepare
$SCRIPT train gpu-recursive-round-v61 0 --web-confirmed --recurrent-depth-ablation
$SCRIPT feedback gpu-recursive-round-v62 --evaluation-confirmed
```

`train` 的第一个参数是**当前累计数据集最后一次已发布的 feedback round ID**，不是这次新训练的 ID。第二个参数是从 EgoLite Active Events 查到的其他 GPU 占用小时；若没有其他 GPU 任务，填 `0`。`--web-confirmed` 表示你已经在 EgoLite 打开原 GPU notebook，并查看 Draft Session 和 Active Events。脚本还会读取官方最新 quota，并把计划分支各自 3 小时一起计入额度。

默认两支是 RLCD depth 1 与 soft-ce depth 1。可在末尾加 `--recurrent-depth-ablation`，在相同训练/留出切分上再加一支 soft-ce depth 2；该对照总计按三支各 3 小时预留。depth 2 的每个 ModernBERT block 使用同一组权重连续计算两次，参数量不增加，编码器计算量约翻倍。该实验是待验证假设，不能预先认为交易准确率会提高。`run.json` 和轮次 summary 会记录 `recurrent_depth`，分支名分别包含 `depth-1` 或 `depth-2`。

`feedback` 的参数是**本次新训练要发布的 feedback round ID**，应使用未用过的序号。示例中，训练以 v61 为输入，冻结评估完成后把结果记为 v62；下一轮训练再传 v62。发布 feedback 前，`--evaluation-confirmed` 要求先在 GPU 日志确认实际 CUDA、所有计划分支完成、冻结评估已经落盘。若包含不同循环深度，feedback 会为每个 loss/depth 组合建立独立策略名，避免 depth 2 覆盖 depth 1 的结果。

脚本会拒绝在固定 kernel 仍 Queued/Running 时再 push，避免重复提交；会要求 Relay 和 CPU 数据准备先完成，GPU 额度不足时停止，不会转 CPU。Relay/CPU/feedback 项目同样复用已有 slug。

## 看状态和统计

| 信息 | 命令/位置 |
| --- | --- |
| GPU 已用、剩余和刷新时间 | `$SCRIPT status` 或 `uvx --from kaggle --with six kaggle quota` |
| 训练实际是否开始、是否完成 | `$SCRIPT status`；再看 Kaggle 网页 Active Events 和 `$SCRIPT logs GPU` |
| 设备、CUDA、分支时长、accuracy、macro-F1、费用后 PnL、冻结留出行数 | `$SCRIPT logs GPU`；以日志里的本轮 JSON summary 为准 |
| Relay 新增 capture、累计归档、坏项和 manifest | `$SCRIPT logs RELAY` |
| CPU 准备训练/候选评估行数与缓存是否命中 | `$SCRIPT logs CPU`；它只准备数据，不做模型微调 |
| feedback 新增/累计行数、有效/无效数和发布状态 | `$SCRIPT logs FEEDBACK` |
| paper-trade / Relay 数据集版本号及版本说明 | Kaggle 对应私有 Dataset 页面 → **Versions**；`status` 显示的文件列表用于辅助看当前文件 |

小留出集上的指标只能描述该轮冻结样本，不能据此声称模型整体改善。每次汇报同时记录训练分支数、每支 GPU 小时、剩余额度、留出行数、累计数据集版本/记录数和下一步。比较循环深度时，重点对比相同 loss、相同数据切分下 depth 1 与 depth 2 的费用后 PnL、macro-F1、balanced accuracy、Brier/ECE，以及运行耗时；小样本只作为流程信号，不作为胜出结论。

## 一轮递归怎么跑

先看当前状态。若 GPU kernel 是 Queued，不要再次 push，也不要仅凭队列推断失败：在 EgoLite 打开**同一个** GPU notebook，检查 Draft Session、Session Metrics 和 Active Events。UI 中的 T4 标签也不等于 Python 进程拿到了 CUDA；训练日志必须显示 CUDA 设备。当前项目曾出现 UI 选中 T4×2、但 Python 是 CPU-only torch 且没有 `/dev/nvidia0` 的情况。

前一阶段完成后按顺序调用：

1. `$SCRIPT relay`，等待 Relay kernel 完成；用 `$SCRIPT logs RELAY` 看是否有新 capture 和新数据集版本。
2. `$SCRIPT prepare`，等待 CPU 数据准备完成；用 `$SCRIPT logs CPU` 看候选评估行。无新鲜时间窗时，不重复使用旧 held-out。
3. 打开 EgoLite 的固定 GPU notebook，检查 Draft Session/Active Events，并确认没有其他 GPU 预留。调用 `$SCRIPT train PREVIOUS_FEEDBACK_ROUND OTHER_GPU_HOURS --web-confirmed`。
4. 若 Queued，回网页检查该事件；重复调用 `status`/`logs GPU` 观察同一版本。直到完成后，从日志确认实际 CUDA、RLCD 和 soft-ce 都结束、冻结指标和留出标签已固定。
5. 调用 `$SCRIPT feedback NEW_FEEDBACK_ROUND --evaluation-confirmed`，等待 feedback kernel 完成；用 `$SCRIPT logs FEEDBACK` 和 Kaggle Dataset → Versions 确认新版本及累计行数。
6. 只有新留出窗口已经成熟、且不会泄漏回上一轮评估时，才用刚发布的 `NEW_FEEDBACK_ROUND` 开始下一次 `train`。

不要对同一个 Queued/Running 版本重复 push，不要用 CPU 做模型微调。只有官方 GPU quota 耗尽或 Kaggle 有明确的 CUDA 分配失败证据，才重新决定剩余 GPU 任务的处理方式；本脚本不会自动回退。

## 周期循环

周期触发适合用于**定时查看状态**，不要把完整训练无条件放进 cron：新 capture、成熟 outcome、未使用的留出窗口和 GPU Session 状态都需要先确认。可每 30 分钟记一次状态：

```cron
*/30 * * * * cd /Users/lin/project/td-market-data-relay && offline_rl/scripts/laya-kaggle-iteration.sh status >> /tmp/laya-kaggle-status.log 2>&1
```

也可在终端持续观察：

```sh
while true; do
  offline_rl/scripts/laya-kaggle-iteration.sh status
  sleep 1800
done
```

实际递归周期仍按上面的阶段顺序推进；GPU 完成并冻结评估、feedback 新版发布、标签成熟后，再启动下一轮。`cycle-help` 会打印阶段清单。Shell 每次只执行你明确调用的阶段，不会后台自行开始下一轮。
