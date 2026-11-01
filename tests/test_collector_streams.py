import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from collector import (fetch_finalized_funding, read_or_initialize_funding_cursor,
                       websocket_url, write_funding_cursor)


class CollectorStreamTests(unittest.TestCase):
    def test_usdm_research_streams_use_market_route(self):
        self.assertEqual(
            websocket_url("usdm-aggtrades", "BTCUSDT"),
            "wss://fstream.binance.com/market/ws/btcusdt@aggTrade",
        )
        self.assertEqual(
            websocket_url("usdm-klines-1m", "BTCUSDT"),
            "wss://fstream.binance.com/market/ws/btcusdt@kline_1m",
        )
        self.assertEqual(
            websocket_url("spot-klines-1m", "BTCUSDT"),
            "wss://stream.binance.com:9443/ws/btcusdt@kline_1m",
        )

    def test_existing_stream_routes_remain_market_specific(self):
        self.assertIn("/public/ws/btcusdt@depth", websocket_url("usdm-depth", "BTCUSDT"))
        self.assertIn("/market/ws/btcusdt@markPrice@1s", websocket_url("usdm-funding-mark", "BTCUSDT"))
        self.assertIn("/ws/btcusdt@depth", websocket_url("spot-depth", "BTCUSDT"))

    def test_finalized_funding_request_is_future_cursor_bound_and_raw_verified(self):
        class Response:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self, _limit):
                return json.dumps([{"symbol": "BTCUSDT", "fundingTime": 2_000,
                                    "fundingRate": "0.0001", "markPrice": "101.25"}]).encode()

        with patch("collector.urllib.request.urlopen", return_value=Response()) as opener:
            rows, raw, url, status = fetch_finalized_funding("BTCUSDT", 1_000)
        self.assertIn("startTime=1000", url)
        opener.assert_called_once()
        self.assertEqual(rows[0]["fundingTime"], 2_000)
        self.assertEqual(json.loads(raw), rows)
        self.assertEqual(status, 200)

    def test_funding_cursor_never_resets_to_historical_data_after_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "funding-cursor.json"
            with patch("collector.time.time_ns", return_value=1_800_000_000_000_000_000):
                first_query_time, first_time = read_or_initialize_funding_cursor(path)
            self.assertEqual(first_query_time, 1_800_000_000_000)
            write_funding_cursor(path, first_time, 1_800_000_000_123)
            query_time_after_restart, original_start_time = read_or_initialize_funding_cursor(path)
            self.assertEqual(query_time_after_restart, 1_800_000_000_124)
            self.assertEqual(original_start_time, first_time)

    def test_funding_cursor_rejects_responses_before_its_future_only_start(self):
        class Response:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self, _limit):
                return b'[{"symbol":"BTCUSDT","fundingTime":999,"fundingRate":"0.1"}]'

        with patch("collector.urllib.request.urlopen", return_value=Response()):
            with self.assertRaisesRegex(ValueError, "future-only cursor"):
                fetch_finalized_funding("BTCUSDT", 1_000)


if __name__ == "__main__":
    unittest.main()
