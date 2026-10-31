#!/usr/bin/env python3
"""Publish complete, finalized Spot one-minute candles from raw captures."""

from export_usdm_klines_dataset import main


if __name__ == "__main__":
    main("SPOT")
