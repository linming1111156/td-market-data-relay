import unittest

from collector import websocket_url


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

    def test_existing_stream_routes_remain_market_specific(self):
        self.assertIn("/public/ws/btcusdt@depth", websocket_url("usdm-depth", "BTCUSDT"))
        self.assertIn("/market/ws/btcusdt@markPrice@1s", websocket_url("usdm-funding-mark", "BTCUSDT"))
        self.assertIn("/ws/btcusdt@depth", websocket_url("spot-depth", "BTCUSDT"))


if __name__ == "__main__":
    unittest.main()
