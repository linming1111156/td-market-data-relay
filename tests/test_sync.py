import hashlib
import json
import os
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import sync
from collector import SUPPORTED_STREAMS as COLLECTOR_SUPPORTED_STREAMS


class RelayHandler(BaseHTTPRequestHandler):
    index_body = b""
    payload = b""

    def do_GET(self):
        if self.path.startswith("/v1/captures?"):
            body = type(self).index_body
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        body = type(self).payload
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


class SyncTests(unittest.TestCase):
    def test_removes_stale_partial_download_only_after_owner_exit(self):
        with tempfile.TemporaryDirectory() as temporary:
            partial = Path(temporary) / "capture.jsonl.download-987654321"
            partial.write_bytes(b"partial")
            with patch("sync.os.kill", side_effect=ProcessLookupError):
                sync.remove_abandoned_temp(partial)
            self.assertFalse(partial.exists())

    def test_preserves_partial_download_while_owner_process_exists(self):
        with tempfile.TemporaryDirectory() as temporary:
            partial = Path(temporary) / "capture.jsonl.download-987654321"
            partial.write_bytes(b"partial")
            with patch("sync.os.kill", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "still running"):
                    sync.remove_abandoned_temp(partial)
            self.assertEqual(partial.read_bytes(), b"partial")

    def test_continuous_reconciliation_rechecks_the_full_remote_retention(self):
        now_ns = 2_000_000_000_000_000_000
        self.assertEqual(sync.continuous_scan_start(now_ns, 0),
                         now_ns - 73 * 3600 * 1_000_000_000)
        self.assertEqual(sync.continuous_scan_start(now_ns, now_ns - 60_000_000_000),
                         now_ns - 60_000_000_000)

    def test_recovery_accepts_every_configured_collector_stream(self):
        collector_streams = COLLECTOR_SUPPORTED_STREAMS
        self.assertEqual(sync.SUPPORTED_STREAMS, collector_streams)
        payload = b'{"record_type":"capture_stopped"}\n'
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for stream in sorted(collector_streams):
                with self.subTest(stream=stream):
                    output = self.run_sync(directory, self.entry_for(payload, stream=stream), payload)
                    self.assertEqual(len(list((output / stream / "BTCUSDT").glob("*.jsonl"))), 1)

    def run_sync(self, directory: Path, entry: dict, payload: bytes, exit_expected=0):
        RelayHandler.index_body = json.dumps({
            "schema": "td_market_data_relay_index_v1", "captures": [entry],
        }).encode()
        RelayHandler.payload = payload
        server = ThreadingHTTPServer(("127.0.0.1", 0), RelayHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        output = directory / "recovered"
        env = {"TD_RELAY_API_KEY": "x" * 64}
        args = ["sync.py", "--base-url", f"http://127.0.0.1:{server.server_port}",
                "--output", str(output)]
        try:
            with patch.dict(os.environ, env), patch.object(sys, "argv", args), redirect_stdout(StringIO()):
                if exit_expected == 0:
                    sync.main()
                else:
                    with self.assertRaises(SystemExit):
                        sync.main()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
        return output

    def entry_for(self, payload: bytes, path: str | None = None, stream: str = "spot-depth") -> dict:
        filename = "BTCUSDT-20261031T120000Z-0123456789abcdef0123456789abcdef.jsonl"
        relative = path or f"{stream}/BTCUSDT/{filename}"
        digest = hashlib.sha256(payload).hexdigest()
        return {"id": "0123456789abcdef0123456789abcdef", "stream": stream,
                "symbol": "BTCUSDT", "path": relative, "compressed": False,
                "size_bytes": len(payload), "modified_at_unix_ns": 1, "sha256": digest,
                "raw_sha256": digest}

    def test_restores_raw_capture_and_is_idempotent(self):
        payload = b'{"record_type":"capture_stopped"}\n'
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            output = self.run_sync(directory, self.entry_for(payload), payload)
            target = output / "spot-depth/BTCUSDT/BTCUSDT-20261031T120000Z-0123456789abcdef0123456789abcdef.jsonl"
            self.assertEqual(target.read_bytes(), payload)
            self.run_sync(directory, self.entry_for(payload), payload)
            self.assertEqual(target.read_bytes(), payload)

    def test_rejects_path_traversal_without_creating_outside_file(self):
        payload = b"unsafe"
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            entry = self.entry_for(payload, "../outside.jsonl")
            output = self.run_sync(directory, entry, payload, exit_expected=1)
            self.assertFalse((directory / "outside.jsonl").exists())
            self.assertTrue(output.exists())

    def test_download_hash_mismatch_does_not_publish(self):
        expected = b"expected payload"
        downloaded = b"different payload"
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            entry = self.entry_for(expected)
            output = self.run_sync(directory, entry, downloaded, exit_expected=1)
            self.assertFalse(any(output.rglob("*.jsonl")))


if __name__ == "__main__":
    unittest.main()
