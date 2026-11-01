import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import zstandard

from build_usdm_rl_dataset import decode_archive


class UsdmRlDatasetInputTests(unittest.TestCase):
    def test_archive_decoder_checks_compressed_raw_hashes_and_recovery_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture_id = "a" * 32
            archive = root / f"BTCUSDT-20261101T000000Z-{capture_id}.jsonl.zst"
            rows = [
                {"schema": "td_public_market_capture_record", "version": 1,
                 "record_type": "capture_started", "stream": "usdm-aggtrades",
                 "symbol": "BTCUSDT", "recorded_at_unix_ns": 100,
                 "source_revision": "fixture", "source_host_id": "relay-source"},
                {"schema": "td_public_market_capture_record", "version": 1,
                 "record_type": "capture_stopped", "recorded_at_unix_ns": 200},
            ]
            raw = b"".join(json.dumps(row, separators=(",", ":")).encode() + b"\n" for row in rows)
            archive.write_bytes(zstandard.ZstdCompressor().compress(raw))
            raw_hash = hashlib.sha256(raw).hexdigest()
            archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
            recovery = {"schema": "td_market_data_relay_recovery_provenance_v1", "version": 1,
                        "capture_id": capture_id, "origin": "vps_recovered",
                        "recovered_at_unix_ns": 300, "raw_sha256": raw_hash,
                        "relay_host": "relay.example"}
            archive.with_name(archive.name + ".meta.json").write_text(json.dumps({
                "schema": "td_market_data_archive_metadata_v1", "capture_id": capture_id,
                "compressed_sha256": archive_hash, "compressed_size_bytes": archive.stat().st_size,
                "raw_sha256": raw_hash, "raw_size_bytes": len(raw),
                "recovery_provenance": recovery,
            }))
            record = decode_archive(archive, root / "decoded" / archive.name.removesuffix(".zst"))
            self.assertEqual(record["origin"], "vps_recovered")
            self.assertEqual(record["recovered_at_unix_ns"], 300)
            self.assertEqual(Path(record["raw_path"]).read_bytes(), raw)
            self.assertEqual(record["raw_sha256"], raw_hash)

    def test_archive_decoder_rejects_compressed_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture_id = "b" * 32
            archive = root / f"BTCUSDT-20261101T000000Z-{capture_id}.jsonl.zst"
            archive.write_bytes(zstandard.ZstdCompressor().compress(b"{}\n"))
            archive.with_name(archive.name + ".meta.json").write_text(json.dumps({
                "schema": "td_market_data_archive_metadata_v1", "capture_id": capture_id,
                "compressed_sha256": "0" * 64, "compressed_size_bytes": archive.stat().st_size,
                "raw_sha256": "0" * 64, "raw_size_bytes": 3,
            }))
            with self.assertRaisesRegex(RuntimeError, "metadata/hash mismatch"):
                decode_archive(archive, root / "decoded.jsonl")


if __name__ == "__main__":
    unittest.main()
