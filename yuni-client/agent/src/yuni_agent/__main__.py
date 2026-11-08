from __future__ import annotations

import argparse
import sys

from pydantic import ValidationError

from yuni_agent.config import StartupConfig
from yuni_agent.main import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Yuni local Agent sidecar")
    parser.add_argument("--dev", action="store_true", help="use local development defaults")
    args = parser.parse_args()

    try:
        config = StartupConfig.development() if args.dev else _read_startup_config()
    except (ValueError, ValidationError) as exc:
        print(f"Invalid startup configuration: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    try:
        run(config)
    except KeyboardInterrupt:
        pass


def _read_startup_config() -> StartupConfig:
    line = sys.stdin.readline()
    if not line:
        raise ValueError("expected one startup configuration JSON object on stdin")
    return StartupConfig.from_json_line(line)


if __name__ == "__main__":
    main()
