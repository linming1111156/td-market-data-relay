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

    def entry_for(self, payload: bytes, path: str | None = None) -> dict:
        filename = "BTCUSDT-20261031T120000Z-0123456789abcdef0123456789abcdef.jsonl"
        relative = path or f"spot-depth/BTCUSDT/{filename}"
        digest = hashlib.sha256(payload).hexdigest()
        return {"id": "0123456789abcdef0123456789abcdef", "stream": "spot-depth",
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
