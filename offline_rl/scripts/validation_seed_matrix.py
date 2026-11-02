#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from td_market_data_rl.analysis.validation_matrix import create_matrix, verify_matrix


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-evaluate frozen Train-seed policies over fixed Validation seeds.")
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--source-sweep", type=Path, required=True)
    create.add_argument("--historical-inputs", type=Path, required=True)
    create.add_argument("--role-plan", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    create.set_defaults(func=create_matrix)
    verify = commands.add_parser("verify")
    verify.add_argument("--artifact", type=Path, required=True)
    verify.add_argument("--historical-inputs", type=Path, required=True)
    verify.add_argument("--role-plan", type=Path, required=True)
    verify.set_defaults(func=verify_matrix)
    args = parser.parse_args()
    try:
        result = args.func(args)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise SystemExit(f"Validation seed matrix rejected: {error}") from error
    if args.command == "create":
        print(f"Validation seed matrix created: {args.output.resolve()}")
    else:
        print("Validation seed matrix verified by recomputing every cross-seed result")


if __name__ == "__main__":
    main()
