# Laya Kaggle 微调：代码与运行路线

本页按当前仓库实现描述 GPU 微调路径。实时 kernel 状态和额度会变化；每次启动前都必须查询 Kaggle CLI，不要把文档里的历史额度或运行状态当作实时值。原始 capture、准备数据、checkpoint、预测和完整运行报告保留在 Kaggle。

## 当前进度（2026-11-22 14:29 UTC）

- 当前待运行轮次：v50，RLCD 与 soft-ce 两个 GPU 分支。
- Kaggle 官方额度查询：已用 0.07h，剩余 29.93/30h；刷新时间为 2026-11-28 00:00 UTC。
- `minglinl99/laya-gpu-loss-ablation-capture-v50` 状态仍为 `QUEUED`，日志为空。本轮尚未开始微调，尚无本轮模型效果或实际 GPU 训练时长。
- 累计 paper-trade 数据集是私有 version 51，最新数据轮次为 `v105-relay-v49-15m-edge0-utcday-v1`。具体样本数以 Kaggle 内该版本的 `manifest.json` 为准；本机不下载或缓存数据文件。
- `scripts/build_laya_kaggle_model_feedback.py` 已准备在 Kaggle 内核中核验完成分支、冻结评估与成熟 outcome，并发布累计数据集新版本；必须等 v50 有真实 CUDA 训练和冻结评估证据后才能运行。
- 下一轮的训练输入仍需接入新增的累计数据集版本，并以未参与此前训练/评估的时间段做新留出集；在这条递归路径实现前，不得把当前反馈数据集发布描述为闭环已完成。

当前工作队列只有 GPU 主线和必要的 Kaggle 数据版本衔接；不排队任何 CPU 微调或 CPU 基线任务。CPU 相关脚本和旧实验记录是历史材料，不是待办项。只在官方 GPU 额度耗尽，或 Kaggle 明确拒绝 GPU/CUDA 且有运行日志证据时，才考虑 GPU 任务的 CPU 回退。

## 当前实现入口

| 文件 | 用途 |
| --- | --- |
| `scripts/build_laya_kaggle_gpu_loss_ablation.py` | 生成私有 Kaggle T4 notebook 和 metadata；本轮比较 RLCD 与 soft-ce |
| `scripts/laya_market_finetune.py` | 在 Kaggle 校验输入、准备/读取切分、微调 Laya 并评估留出数据 |
| `bulk_engine/src/laya_prepare.rs` | 从 Relay capture 生成时间排序的训练集与评估集 |
| `scripts/build_laya_kaggle_paper_trade_dataset.py` | 从固定的历史 v24 运行构建 v31 paper-trade 回放数据；它不是当前 v50 模型结果的自动追加器，也不负责训练模型 |

## 当前配对实验

生成器的默认目标为 `minglinl99/laya-gpu-loss-ablation-capture-v50`。它创建私有 notebook，启用 GPU 与 Internet，设置 `NvidiaTeslaT4`，并挂载两个冻结的 Relay capture datasets。生成器把当前仓库里的 Rust 源码和微调脚本嵌入 notebook，因此 notebook 运行时不依赖本机工作区。

该轮固定 15 分钟决策间隔、15 分钟标签期限、edge 0 bps、训练比例 80%、encoder 冻结、micro-batch 16、gradient accumulation 8、1 epoch、seed 20261122；只比较 `rlcd` 和 `soft-ce`。每支默认计划上限 1.5 GPU 小时，配对总计划 3 小时。Notebook 检查 Relay manifest SHA、Rust 准备报告 schema、训练 836 条（flat/long/short 为 639/98/99）、评估 203 条（143/29/31）和跳过记录；不匹配就停止。

## 生成、提交和监控

先从官方 CLI 读取新鲜额度，并核对其他运行或排队的 GPU kernel：

```sh
uvx --from kaggle --with six kaggle quota
uvx --from kaggle --with six kaggle kernels status OWNER/KERNEL-SLUG
```

依据本次查询填写已用和其他预留小时。以下 SHA 与参数仅对应冻结的 v50 snapshot；开始新一轮时必须先核实目标 manifest 与额度，不可照抄过期数字。

```sh
python3 offline_rl/scripts/build_laya_kaggle_gpu_loss_ablation.py \
  --output-dir /mnt/data/kaggle-kernels \
  --expected-relay-manifest-sha256 4e84be91787cef52499be3941ede0a77af6f6a517dac4962ec0111e1a01b222c \
  --quota-used-hours FRESH_USED_HOURS_FROM_QUOTA \
  --reserved-gpu-hours VERIFIED_OTHER_GPU_RESERVATIONS

uvx --from kaggle --with six kaggle kernels push \
  -p /mnt/data/kaggle-kernels/laya-gpu-loss-ablation-capture-v50
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

训练结果冻结后才能用 held-out 标签。训练脚本当前输出逐样本预测和真实标签，但仓库里的 v31 paper-trade builder 仍绑定 v24 运行；因此“将 v50 每轮预测与成熟真实 outcome 追加到累计数据集新版本，再以该版本启动下一轮”的端到端步骤尚未由现有脚本自动完成。文档和汇报必须明确这一实现边界，不能把 v31 数据集构建描述成 v50 反馈闭环已经完成。

运行数据仅留 Kaggle。用 bounded logs 查看运行进度与摘要；不要下载 notebook outputs、原始行情、准备后的样本或 checkpoint 到本机。

## 汇报内容

每轮汇报分支数量、已开始/完成/失败数、实际 GPU 小时和剩余额度、评估指标、累计数据集版本与记录数、下一步。超过 10 分钟的运行至少每 10 分钟更新一次；训练尚未开始时明确说“未完成微调”，不填写推测效果。

## 递归迭代顺序

1. 在 Kaggle GPU 上完成当前配对微调，并保存 checkpoint、逐样本预测、真实 outcome、运行清单和冻结评估结果。
2. 核对模型预测、成熟 outcome、数据时间戳与文件哈希；只把已验证结果追加到 Kaggle 累计数据集的新私有版本。
3. 下一轮读取刚发布的数据集版本训练，并使用此前未进入训练或评估的后续时间段作为留出集；先冻结新一轮输入与切分，再开始 GPU 微调。
4. 每轮结束更新本页中的轮次、运行状态、GPU 额度、指标、数据集版本及记录数，然后提交并推送代码和文档。

若当前 kernel 仍为 `QUEUED`，继续监控同一 kernel；不得重复提交同一轮，也不得把等待当成训练完成。额度充足时继续排队/运行 GPU 主线，不启动 CPU 微调。
