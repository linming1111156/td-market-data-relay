import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import zstandard

from capture_recovery import SCHEMA, recovery_provenance
from scripts.stage_verified_capture_archive import stage_capture


class StageVerifiedCaptureArchiveTests(unittest.TestCase):
    def test_staging_restores_vps_recovery_sidecar_from_archive_metadata(self):
        payload = b'{"record_type":"capture_stopped"}\n'
        raw_sha = hashlib.sha256(payload).hexdigest()
        capture_id = "0123456789abcdef0123456789abcdef"
        filename = f"BTCUSDT-20261106T000000Z-{capture_id}.jsonl"
        recovery = {
            "schema": SCHEMA,
            "version": 1,
            "capture_id": capture_id,
            "origin": "vps_recovered",
            "recovered_at_unix_ns": 123456789,
            "raw_sha256": raw_sha,
            "relay_host": "relay.example",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hot = root / "hot"
            archive = root / "archive"
            staging = root / "staging"
            hot.mkdir()
            packed = archive / "spot-depth/BTCUSDT" / f"{filename}.zst"
            packed.parent.mkdir(parents=True)
            compressed = zstandard.ZstdCompressor().compress(payload)
            packed.write_bytes(compressed)
            metadata = {
                "schema": "td_market_data_archive_metadata_v1",
                "capture_id": capture_id,
                "raw_sha256": raw_sha,
                "raw_size_bytes": len(payload),
                "compressed_sha256": hashlib.sha256(compressed).hexdigest(),
                "compressed_size_bytes": len(compressed),
                "recovery_provenance": recovery,
            }
            packed.with_name(packed.name + ".meta.json").write_text(json.dumps(metadata))

            staged = stage_capture(hot / "spot-depth/BTCUSDT" / filename, hot, archive, staging)

            self.assertEqual(staged.read_bytes(), payload)
            self.assertEqual(recovery_provenance(staged, raw_sha, capture_id), recovery)

    def test_invalid_archive_provenance_is_rejected_before_staging(self):
        payload = b'{"record_type":"capture_stopped"}\n'
        raw_sha = hashlib.sha256(payload).hexdigest()
        capture_id = "0123456789abcdef0123456789abcdef"
        filename = f"BTCUSDT-20261106T000000Z-{capture_id}.jsonl"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hot = root / "hot"
            archive = root / "archive"
            staging = root / "staging"
            hot.mkdir()
            packed = archive / "spot-depth/BTCUSDT" / f"{filename}.zst"
            packed.parent.mkdir(parents=True)
            compressed = zstandard.ZstdCompressor().compress(payload)
            packed.write_bytes(compressed)
            metadata = {
                "schema": "td_market_data_archive_metadata_v1",
                "capture_id": capture_id,
                "raw_sha256": raw_sha,
                "raw_size_bytes": len(payload),
                "compressed_sha256": hashlib.sha256(compressed).hexdigest(),
                "compressed_size_bytes": len(compressed),
                "recovery_provenance": {"origin": "vps_recovered"},
            }
            packed.with_name(packed.name + ".meta.json").write_text(json.dumps(metadata))

            with self.assertRaisesRegex(ValueError, "recovery provenance"):
                stage_capture(hot / "spot-depth/BTCUSDT" / filename, hot, archive, staging)
            self.assertFalse((staging / "spot-depth/BTCUSDT" / filename).exists())


if __name__ == "__main__":
    unittest.main()
