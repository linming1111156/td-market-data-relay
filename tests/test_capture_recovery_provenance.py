import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from capture_recovery import SCHEMA, capture_id_for, recovery_provenance
from sync import write_recovery_provenance


class CaptureRecoveryProvenanceTests(unittest.TestCase):
    def test_sync_writes_idempotent_hash_bound_recovery_record(self):
        with tempfile.TemporaryDirectory() as directory:
            capture = Path(directory) / ("BTCUSDT-20261101T000000Z-" + "a" * 32 + ".jsonl")
            capture.write_bytes(b"capture\n")
            raw_hash = hashlib.sha256(capture.read_bytes()).hexdigest()
            entry = {"id": "a" * 32, "raw_sha256": raw_hash}
            write_recovery_provenance(capture, entry, "relay.example")
            sidecar = capture.with_name(capture.name + ".recovery.json")
            first = json.loads(sidecar.read_bytes())
            write_recovery_provenance(capture, entry, "relay.example")
            self.assertEqual(json.loads(sidecar.read_bytes()), first)
            self.assertEqual(first["schema"], SCHEMA)
            self.assertEqual(recovery_provenance(capture, raw_hash, "a" * 32), first)

    def test_recovery_sidecar_must_match_capture_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            capture = Path(directory) / ("BTCUSDT-20261101T000000Z-" + "b" * 32 + ".jsonl")
            capture.write_bytes(b"capture\n")
            raw_hash = hashlib.sha256(capture.read_bytes()).hexdigest()
            capture_id = capture_id_for(capture, raw_hash)
            sidecar = capture.with_name(capture.name + ".recovery.json")
            sidecar.write_text(json.dumps({
                "schema": SCHEMA, "version": 1, "capture_id": capture_id,
                "origin": "vps_recovered", "recovered_at_unix_ns": 123,
                "raw_sha256": "0" * 64, "relay_host": "relay.example",
            }))
            with self.assertRaises(ValueError):
                recovery_provenance(capture, raw_hash, capture_id)


if __name__ == "__main__":
    unittest.main()
