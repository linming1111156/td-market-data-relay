# Laya 市场微调迭代路线

> 执行入口只有“当前状态”和“GPU 优先迭代闭环”。历史结果只作证据，不是待办任务。原始市场数据和完整运行产物只留在 Kaggle；本机只保存代码、skill 和路线图。

## 当前状态（2026-11-22）

- **正在进行的 GPU 轮**：capture v50 的 RLCD/soft-ce 配对微调。冻结 15m/edge0 切分，encoder 冻结、MB16、seed 20261122、1 epoch、grad-accum 8，只改变 loss。version 1 在 Rust 准备后因报告字段校验错误终止，两个微调分支均未开始；官方 GPU 已用 0.07h、剩余 29.93/30h。修正后的 version 2 当前状态为 QUEUED；尚无本轮训练结果或 CUDA 证据。
- **历史微调数量和效果**：有记录的已训练分支共 15 个：初始/修复 MB4、MB8 共 3 个，v8 MB16/MB32 共 8 个，v24 loss ablation 共 4 个。MB4/MB8 的旧切分 accuracy 为 14.67%，校准 loss 有改善但未证明费用后交易效果；之后 12 个分支在旧 150 条评估集都预测 short，accuracy 14.67%、macro-F1 8.53%、费用后净 PnL -4,288.825 bps，均未晋级。旧数据指标不能代替 capture v50 本轮评估。
- **当前数据数量**：私有 capture version 50 有 442 个归档通过完整性校验，重复 ID 为 0、缺 sidecar 为 0；finalized-funding 累计 6,171 次 HTTPError、成功页为 0。标签矩阵 v104 的冻结 15m/edge0 切分是训练 836 条（flat/long/short 为 639/98/99）、评估 203 条（143/29/31）。累计经验集 version 51 有 1,022 条决策（1,007 valid、15 invalid）、1,187 个质量事件、350 个亏损 outcome 保留；上下文覆盖 1,021/1,022。
- **本轮下一步**：等 v50 kernel 离开 QUEUED，核对实际 T4/CUDA；两个 loss 分支完成后记录训练效果、费用后结果、分时段结果和 GPU 小时。冻结指标后，把成熟真实 outcome 追加到 Kaggle 累计数据集新版本，并留存 checkpoint 和 run manifest；完成入库后再开下一轮 GPU 微调。

## GPU 优先迭代闭环（唯一待做主线）

1. **冻结输入和假设**：选定 Kaggle 私有数据版本，记录 Relay manifest SHA、时间切分、训练/校准/评估样本数、起始 checkpoint，并在每轮只改变一个因素。
2. **先启动 GPU 微调**：每轮启动前用官方 Kaggle CLI 查询 7 天额度、检查在跑/排队 GPU 项目；有额度就显式请求 T4 并核验真实 CUDA。官方额度未归零时，不得把 GPU 兼容模型训练或推理改排到 CPU。只有额度归零，或 Kaggle 明确拒绝分配 GPU/CUDA 不可用并留有日志证据，才可 CPU 回退。不得超过官方滚动 30 小时上限，也不得重复实验只为消耗额度。
3. **保存每个模型和运行报告**：checkpoint、输入/准备数据 SHA、loss、seed、训练配置、实际设备、GPU 小时、训练指标、预测和 run manifest 都保留在 Kaggle；每个分支单独标注成功、失败或未开始。
4. **在冻结评估集验收**：报告 accuracy、balanced accuracy、macro-F1、混淆矩阵、校准、交易次数、费用/滑点后 PnL、回撤和每个未见时间段结果。held-out 结果只用于本轮评估，不用于本轮调阈值或提前训练。
5. **把真实运行结果追加到累计数据集**：评估结论冻结后，将逐样本预测、样本/模型/轮次 ID、成熟真实标签、成交与费用、质量状态追加到 Kaggle 私有累计数据集的新版本；不覆盖旧版本。只用真实已实现 outcome 作为训练标签，严禁把模型预测当真值。亏损、无效、未决样本保留并标记；当前 held-out 标签只在本轮指标冻结后才可供下一轮使用。
6. **启动下一轮 GPU 微调**：下一轮读取上轮累计数据集和选定 checkpoint，冻结新的输入切分、提出可证伪假设，再从第 2 步开始。必要的 Rust/数据准备可以作为该 Kaggle notebook 内的支持步骤，不单独排队 CPU 研究任务。

## 汇报要求

任务提交、GPU 分配确认、每个微调分支完成、累计数据集发版和下一轮启动时向用户汇报；运行超过 10 分钟时至少每 10 分钟汇报一次。每次说明微调轮次/分支数、实际 GPU 小时与余额、模型微调后的效果、当前累计数据集数量、下一步方向。失败前没有执行优化器的任务明确写为“未完成微调”。

## Kaggle CLI / MCP 使用

- 查官方滚动额度：uvx --from kaggle --with six kaggle quota。此环境的 CLI 会使用 ~/.kaggle/access_token；不得读取、输出或提交 token 内容，也不需要网页登录。
- Kaggle MCP 用于其 schema 明确支持的数据集列表/元数据/文件与 notebook 信息操作。当前 MCP 没有 quota 工具，kernel_push 也不支持 metadata/accelerator；GPU notebook 使用官方 CLI。
- GPU notebook 使用 kernel-metadata.json 显式设置私有性、数据集 sources、Internet 和 GPU；通过 CLI 指定 accelerator，并用 kernel status/logs 核验。日志接口不可用时要等到终态再记录指标，不能猜测 CUDA 或效果。
- 不下载 Kaggle notebook outputs、原始行情、派生样本或 checkpoint 到本机。

## 重要历史对照（已完成，不是待办）

- capture version 50 覆盖审计：442 个归档全部有效，无重复 ID、无缺 sidecar；finalized-funding 没有成功 API 页。
- label sensitivity v104 和只读索引均 COMPLETE / pass，9/9 配置通过；冻结 15m/edge0 的训练/评估数量及类别见当前状态。
- capture version 49 CPU 基线 v107：trend、reversal、balanced logistic 的费用后净 PnL 分别为 -824.281、-2,835.719、-4,317.353 bps；六个连续 session 切片均无正净值段，不晋级。仅用于历史对照。
- 2026-11-22 Kaggle quota 查询：GPU 0.07h 已用、29.93h 剩余、总额 30h；刷新时间 2026-11-28 00:00 UTC。该值随任务变化，不能替代下一次启动前的新鲜查询。

## 数据与评估边界

- 行情只从 Kaggle 私有 capture datasets 读取；Relay 原始输出、完整报告、checkpoint、预测文件和运行数据均留 Kaggle。
- 决策时点只能使用当时已知特征；先预测，再等待未来窗口完成并落真实 outcome。每条记录带稳定 sample ID、round ID、source snapshot、model/version、输入哈希、预测/概率、标签、费用、交易与数据质量状态。
- 有效性与盈亏分开：亏损但完整可信的样本仍保留；无效、缺失和未成熟样本保留并标明原因，不静默删除。保留原始 dataset versions。
- 现有 capture snapshot 的市场时间跨度和独立行情段有限，finalized-funding 当前不可用；本轮结果是历史回放诊断，不宣称生产收益或跨市场泛化。
