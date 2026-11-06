# Local VPS sync storage

The local sync service stores verified VPS captures in
`/mnt/data/td-market-data-local/archive` as `.jsonl.zst` files with relay-compatible
`.meta.json` sidecars. It does not persist recovered capture bodies under `hot/`.

For remote archived captures, sync verifies the downloaded compressed SHA-256,
streams decompression through a bounded JSONL validator, checks the raw SHA-256,
and confirms the capture ends with `capture_stopped`. It then publishes the
original compressed bytes. For recent remote hot captures that are still raw,
sync validates them and compresses them once into the archive. Metadata records
the compressed and raw hashes and VPS recovery provenance.

The local relay can serve these archives and applies the existing 73-hour
retention policy. Offline dataset jobs can stage one verified archive into a
separate working directory with `scripts/stage_verified_capture_archive.py`;
staged JSONL is an input artifact for a job, not the durable capture store. The
stager validates and recreates the hash-bound recovery sidecar from archive
metadata so point-in-time dataset exporters retain VPS recovery timestamps.

The local capture services remain disabled on the development host. The sync
unit uses `--direct-archive`; its cursor and remote-loss ledger remain under
`/mnt/data/td-market-data-local/state`.

To audit an archive tree without extracting files to disk, build the Rust bulk
engine and run its bounded parallel verifier:

```sh
cargo build --release --manifest-path bulk_engine/Cargo.toml
bulk_engine/target/release/td-market-data-bulk-engine verify-archives \
  --archive-root /mnt/data/td-market-data-local/archive --workers 8
```

The verifier checks each archive's compressed and raw SHA-256 and any byte
counts recorded in its sidecar, parses its JSONL records, and requires the
final record to be `capture_stopped`. It accepts the standard archive metadata
sidecar and the recovery provenance sidecar used by the earlier VPS recovery
folder. Worker count is explicit and bounded to 1–32 so operators can limit
CPU and I/O pressure.
