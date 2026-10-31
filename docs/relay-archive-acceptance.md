# Relay archive and local recovery acceptance

The local relay and sync paths were exercised without an external VPS or public
network. A temporary local HTTP server used the real relay handler and real
`zstandard` package; test data lived in a temporary directory and was removed
after each case.

Checks:

- capture compression preserves the raw bytes, emits zstd, and records matching
  compressed and raw SHA-256 values;
- the archive API denies unauthenticated calls and returns captures with a
  correct bearer key;
- `sync.py` restores the compressed capture, verifies both hashes, and writes
  the original raw JSONL bytes;
- an active collector file lock prevents rotation;
- the 73-hour expiry deletes capture and metadata together;
- archive quota eviction deletes the oldest capture and its metadata together;
- malformed path and download hash mismatch tests refuse publication.

The full suite reports 11 passing tests. This validates local behavior only;
Docker Compose, Cloudflare Tunnel, and the future local systemd unit have not
been deployed because no VPS/relay endpoint is provisioned.
