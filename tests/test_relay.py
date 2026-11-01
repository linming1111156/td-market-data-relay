import fcntl
import hashlib
import json
import os
import tempfile
import threading
import time
import urllib.error
import urllib.request
import unittest
import uuid
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import zstandard

import relay
import sync


def write_capture(root: Path, start_seconds: float | None = None) -> tuple[Path, bytes]:
    root.mkdir(parents=True, exist_ok=True)
    capture_id = uuid.uuid4().hex
    captured_at = start_seconds if start_seconds is not None else time.time() - 7200
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(captured_at))
    path = root / "usdm-depth" / "BTCUSDT" / f"BTCUSDT-{stamp}-{capture_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"schema": "td_public_market_capture_record", "version": 1,
         "record_type": "capture_started", "recorded_at_unix_ns": int(captured_at * 1e9),
         "stream": "usdm-depth", "symbol": "BTCUSDT"},
        {"schema": "td_public_market_capture_record", "version": 1,
         "record_type": "capture_stopped", "recorded_at_unix_ns": int((captured_at + 3600) * 1e9)},
    ]
    payload = b"".join((json.dumps(row, separators=(",", ":")) + "\n").encode() for row in rows)
    path.write_bytes(payload)
    os.utime(path, (captured_at + 3600, captured_at + 3600))
    return path, payload


class QuietRelayHandler(relay.Handler):
    def log_message(self, *_args):
        pass


class RelayTests(unittest.TestCase):
    def test_archive_round_trip_auth_and_sync(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            hot, archive, output = base / "hot", base / "archive", base / "recovered"
            path, original = write_capture(hot)
            capture_id = path.name.split("-")[-1].split(".")[0]
            api_key = "k" * 64
            with ExitStack() as stack:
                stack.enter_context(patch.object(relay, "HOT", hot))
                stack.enter_context(patch.object(relay, "ARCHIVE", archive))
                stack.enter_context(patch.object(relay, "API_KEY", api_key))
                stack.enter_context(patch.object(relay, "MAX_ARCHIVE_BYTES", 1024 * 1024))
                stack.enter_context(patch.object(relay, "MAX_HOT_BYTES", 1024 * 1024))
                relay.compress_old_captures()
                self.assertFalse(path.exists())
                compressed = archive / "usdm-depth" / "BTCUSDT" / (path.name + ".zst")
                metadata_path = compressed.with_name(compressed.name + ".meta.json")
                metadata = json.loads(metadata_path.read_text())
                self.assertEqual(metadata["raw_sha256"], hashlib.sha256(original).hexdigest())
                with compressed.open("rb") as source:
                    with zstandard.ZstdDecompressor().stream_reader(source) as reader:
                        self.assertEqual(reader.read(), original)
                server = relay.ThreadingHTTPServer(("127.0.0.1", 0), QuietRelayHandler)
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                try:
                    base_url = f"http://127.0.0.1:{server.server_port}"
                    with self.assertRaises(urllib.error.HTTPError) as error:
                        urllib.request.urlopen(base_url + "/v1/captures")
                    self.assertEqual(error.exception.code, 401)
                    with patch.dict(os.environ, {"TD_RELAY_API_KEY": api_key}), \
                            patch.object(sync, "request", wraps=sync.request), \
                            patch("sys.argv", ["sync.py", "--base-url", base_url,
                                               "--output", str(output)]):
                        sync.main()
                    restored = output / "usdm-depth" / "BTCUSDT" / path.name
                    self.assertEqual(restored.read_bytes(), original)
                    index = relay.capture_index(0, time.time_ns(), "", "")
                    self.assertEqual([item["id"] for item in index], [capture_id])
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=2)

    def test_active_capture_lock_prevents_rotation_and_expired_archive_removes_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            hot, archive = base / "hot", base / "archive"
            active, _ = write_capture(hot)
            with active.open("rb") as stream:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                with patch.object(relay, "HOT", hot), patch.object(relay, "ARCHIVE", archive):
                    relay.compress_old_captures()
            self.assertTrue(active.exists())
            expired = archive / "usdm-depth" / "BTCUSDT" / "expired-0123456789abcdef0123456789abcdef.jsonl.zst"
            expired.parent.mkdir(parents=True, exist_ok=True)
            expired.write_bytes(b"old archive")
            metadata = expired.with_name(expired.name + ".meta.json")
            metadata.write_text("{}\n")
            old_time = time.time() - 74 * 3600
            os.utime(expired, (old_time, old_time))
            with patch.object(relay, "HOT", hot), patch.object(relay, "ARCHIVE", archive):
                relay.compress_old_captures()
            self.assertFalse(expired.exists())
            self.assertFalse(metadata.exists())

    def test_archive_quota_evicts_oldest_and_its_metadata_together(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            hot, archive = base / "hot", base / "archive"
            old = archive / "usdm-depth" / "BTCUSDT" / "BTCUSDT-20261031T120000Z-0123456789abcdef0123456789abcdef.jsonl.zst"
            old.parent.mkdir(parents=True, exist_ok=True)
            old.write_bytes(b"x" * 450)
            metadata = old.with_name(old.name + ".meta.json")
            metadata.write_text("{}\n")
            old_time = time.time() - 2 * 3600
            os.utime(old, (old_time, old_time))
            write_capture(hot)
            with patch.object(relay, "HOT", hot), patch.object(relay, "ARCHIVE", archive), \
                    patch.object(relay, "MAX_ARCHIVE_BYTES", 500):
                relay.compress_old_captures()
                losses = relay.capture_losses(0, time.time_ns(), "", "")
            self.assertFalse(old.exists())
            self.assertFalse(metadata.exists())
            self.assertEqual(len(list(archive.rglob("*.jsonl.zst"))), 1)
            self.assertEqual(len(losses), 1)
            self.assertEqual(losses[0]["capture_id"], "0123456789abcdef0123456789abcdef")
            self.assertEqual(losses[0]["reason"], "archive_quota")

    def test_archive_quota_refuses_unrecorded_eviction_when_loss_ledger_is_full(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            hot, archive = base / "hot", base / "archive"
            old = archive / "usdm-depth" / "BTCUSDT" / "BTCUSDT-20261031T120000Z-0123456789abcdef0123456789abcdef.jsonl.zst"
            old.parent.mkdir(parents=True, exist_ok=True)
            old.write_bytes(b"x" * 450)
            metadata = old.with_name(old.name + ".meta.json")
            metadata.write_text("{}\n")
            old_time = time.time() - 2 * 3600
            os.utime(old, (old_time, old_time))
            incoming, _ = write_capture(hot)
            with patch.object(relay, "HOT", hot), patch.object(relay, "ARCHIVE", archive), \
                    patch.object(relay, "MAX_ARCHIVE_BYTES", 500), \
                    patch.object(relay, "MAX_LOSS_LEDGER_BYTES", 1):
                relay.compress_old_captures()
            self.assertTrue(old.exists())
            self.assertTrue(incoming.exists())

    def test_hot_storage_quota_records_removed_raw_capture(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            hot, archive = base / "hot", base / "archive"
            capture, _ = write_capture(hot, start_seconds=time.time())
            with patch.object(relay, "HOT", hot), patch.object(relay, "ARCHIVE", archive), \
                    patch.object(relay, "MAX_HOT_BYTES", 1):
                relay.enforce_hot_cap()
                losses = relay.capture_losses(0, time.time_ns(), "", "")
            self.assertFalse(capture.exists())
            self.assertEqual(len(losses), 1)
            self.assertEqual(losses[0]["reason"], "hot_storage_quota")


if __name__ == "__main__":
    unittest.main()
