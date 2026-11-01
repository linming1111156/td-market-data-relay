# Upstream source availability — 2026-11-01

## Binance egress restriction observed from the relay host

A read-only request from the relay Docker network to each public time endpoint
returned HTTP 451:

- `https://api.binance.com/api/v3/time`
- `https://fapi.binance.com/fapi/v1/time`

Both responses included Binance's message that service is unavailable from a
restricted location under section `b. Eligibility` of its
[Terms](https://www.binance.com/en/terms). This establishes that Binance
classified the relay host's egress as a restricted location for these
requests. It does not by itself establish which location database or network
property caused that classification.

The affected relay captures are not research inputs. A successful download,
matching file hash, healthy relay endpoint, or Cloudflare Tunnel connection
only proves transport and storage integrity; it does not turn a capture with
zero market frames or HTTP 451 failures into valid market data. Keep those
failure records for operational diagnosis and exclude them from Dataset
publication and replay.

Do not route these requests through another region to evade the restriction.
If Binance believes the classification is incorrect, follow the response and
contact Binance support. Otherwise use a market-data source that is authorized
for the operator's location and offers the required instruments. Confirm its
availability before provisioning additional collector nodes.

## Research collection status

The local collector remains a separate path and has successfully received
USD-M Mark Price/Funding observations and public Funding History responses.
Those local captures must still pass the existing lifecycle, hash, gap,
settlement, and execution-replay gates before they can support a research
conclusion. No data from frozen historical Train/Validation/Test/OOS
partitions is involved.
