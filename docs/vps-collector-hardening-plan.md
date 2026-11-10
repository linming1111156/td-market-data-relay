# VPS collector hardening plan

更新时间：2026-11-10

## 当前诊断

采集器已经有断线重连循环。异常后会按 1、2、4 秒退避，最长 60 秒再建立 WebSocket；深度流重连时会重新获取 REST 快照并重新桥接序列。

当前主要缺陷是缺口处理：aggTrades 在收到不连续的交易 ID 后，会记录 `gap_detected`，将高水位前移，然后继续写入同一个分片。重连只能接收新数据，不能补回断线期间已经错过的交易，因此这种分片必须被标记为不可训练，或者先用 REST 按 ID 补齐。

2026-11-10 对 VPS 同步数据的检查结果：43 个 USD-M 深度分片中 25 个无缺口；43 个 USD-M aggTrades 分片中只有 5 个无缺口。文件哈希和 zstd 校验正常，缺口产生在 VPS WebSocket 采集阶段，不是同步传输损坏。

## 任务清单

### Phase 1：可靠性修复

- [x] aggTrades 第一个交易 ID 缺口出现时立即结束当前逻辑分片，禁止继续把不连续数据伪装成连续数据。当前实现关闭分片并以 `reason=sequence_gap` 隔离，然后重置高水位建立新分片。
- [ ] 记录缺口的前后交易 ID、会话 ID、断线原因和重连次数。
- [ ] 评估 Binance REST aggTrades 按交易 ID补回；补回不完整时保留不可训练标记并丢弃该窗口。
- [x] 深度流继续采用“断线、重新快照、重新桥接”的流程；任何桥接失败都让当前分片以 `sequence_gap` 隔离。
- [ ] relay index 明确暴露 `gap_count`、`disconnect_count` 和可训练状态，避免本地同步后才发现不可用。
- [ ] 增加 VPS 端健康指标：连接数、重连数、缺口数、队列溢出、接收延迟和分片淘汰数。

### Phase 2：性能基准

- [ ] 在 VPS 上分别测量 WebSocket 接收、JSON 解析、SHA-256、JSONL 写盘、fsync 和 relay 压缩的 CPU、RSS、磁盘写入量。
- [ ] 使用 VPS 真实运行窗口记录队列深度和 `receiver queue overflow`，不要只用离线文件吞吐推断瓶颈。
- [ ] 以“无队列溢出、无因 CPU 饱和造成的接收延迟、RSS 不超过容器限制”为验收条件。

仓库提供 `scripts/profile_collector_hotpath.py` 作为本地热路径基准。当前开发机用 1,000 个合成帧测得约 37,416 帧/秒的完整 JSONL 写入路径；这只是合成基线，不代表 VPS 实际吞吐，必须在 VPS 运行窗口再次测量。

新镜像包含该脚本，可在 VPS 使用共享数据卷运行：

```sh
docker compose exec -T usdm-aggtrades \
  python3 /app/scripts/profile_collector_hotpath.py \
  --frames 10000 --work-dir /var/lib/td-market-data/state
```

### Phase 3：Rust 加速边界

- [ ] 只有当基准证明 Python 的 JSON 解析、字段校验或哈希占用主要 CPU 时，才实现 Rust 辅助引擎。
- [ ] 候选 Rust 热点：原始帧字段校验、交易/深度序列检查、批量 JSONL 封装和 SHA-256。
- [ ] 保留 Python 负责 WebSocket、SOCKS5、退避重连、REST 补回和策略编排；Rust 不负责网络恢复。
- [ ] 优先采用独立 Rust helper 或稳定的批处理接口，避免为单帧调用引入 PyO3 跨语言开销。
- [ ] Rust 版本必须与 Python 版本逐条比对哈希、序列结果、gap 标记和输出字节，再进入镜像。

当前静态热点位于 `collector.py` 的每帧 `json.loads`、原始帧 UTF-8 编码、SHA-256、`json.dumps` 和无缓冲写盘；网络接收、SOCKS5 和退避重连仍属于 Python 编排层。尚未有 VPS 运行时 CPU、队列溢出或 RSS 基准，因此暂不引入 Rust helper。

## 暂不做的事情

- 不因为存在缺口就盲目把整个采集器重写成 Rust。
- 不用 Rust 试图修复 VPS 代理或网络丢包；这属于连接恢复和数据回补问题。
- 不把带 gap 的分片送入 Laya 或 RL 训练。

## 完成标准

1. 连续运行期间，健康分片 `gap_count=0`，或者缺口分片被明确隔离并有可追踪原因。
2. aggTrades 可回补的缺口经过交易 ID 和哈希校验；不可回补的窗口不会进入训练集。
3. 深度重连后的快照桥接通过序列校验。
4. 性能基准证明 Rust 改动带来实际收益后，才合入 Rust helper；否则保持 Python 实现。

## 部署说明

本次改动只影响 collector 镜像和测试/文档，不影响正式 Gateway。发布镜像后，在 VPS 项目目录执行：

```sh
git pull --ff-only
docker compose config --quiet
docker compose pull
docker compose up -d --no-build
docker compose ps
docker compose logs --since=5m --tail=100 usdm-aggtrades usdm-depth
```

重点观察新分片是否出现 `capture segment discarded: ... reason=sequence_gap`，以及随后建立的分片是否有 `gap_count=0`。带 `reason=sequence_gap` 的旧分片必须保留为审计证据，但不得进入训练集。
