# Spot execution replay acceptance

`spot_execution_replay.py` verifies the Spot Dataset file hashes and logical
fingerprint, checks the update-ID chain again, and replays bounded
`MARKET_IOC` intents in timestamp order. An order sees no market event until
both its exchange event timestamp and local receive timestamp make it
available. Simulated fills use the visible arrival-time L2 book and consume
that liquidity within the account replay. Buys are limited by quote balance
including taker fees; sells are limited by base inventory. Unfilled IOC
remainder is reported as a partial or unfilled order.

The report contains quote-denominated equity, net P&L, fees, turnover, and
adverse cost versus decision-time mid. Strategy, NOOP, and Buy-and-Hold accounts
share initial balances, terminal midpoint, configured fee, and latency. The
Buy-and-Hold benchmark attempts one market buy at the window opening and then
holds; visible depth and cash limits still apply.

Synthetic acceptance covers partial fills, fee arithmetic, turnover, same-ledger
NOOP and Buy-and-Hold accounting, create-only report output, and rejection of
tampered Dataset files. Repeated runs with identical inputs produce identical
reports; each report binds the Dataset manifest, config, orders, and replay
code hashes. This does not validate a live capture or establish economic
usefulness. Queue position, hidden liquidity, maker fills, Funding, and unified
Spot/USD-M portfolio comparison remain outside the current model.
