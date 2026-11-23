
"""Sample Docker container resource usage during load testing."""

import argparse
import csv
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Monitor CPU and memory usage of a Docker container."
    )
    parser.add_argument(
        "--container",
        default="mibici-api-baseline",
        help="Docker container name or ID.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Sampling interval in seconds.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination CSV file.",
    )
    args = parser.parse_args()

    if args.interval <= 0:
        parser.error("--interval must be greater than zero")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "timestamp_utc",
        "container",
        "cpu_percent",
        "memory_usage",
        "memory_limit",
        "memory_percent",
        "network_io",
        "block_io",
        "pids",
    ]

    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        file.flush()

        print(
            f"Monitoring {args.container} every {args.interval}s. "
            "Press Ctrl+C to stop.",
            flush=True,
        )

        try:
            while True:
                result = subprocess.run(
                    [
                        "docker",
                        "stats",
                        "--no-stream",
                        "--format",
                        "{{json .}}",
                        args.container,
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )

                if not result.stdout.strip():
                    raise RuntimeError("Docker returned no container metrics.")

                stats = json.loads(result.stdout.strip().splitlines()[0])

                writer.writerow(
                    {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "container": args.container,
                        "cpu_percent": float(stats["CPUPerc"].rstrip("%")),
                        "memory_usage": stats["MemUsage"],
                        "memory_limit": stats["MemUsage"].split(" / ")[-1],
                        "memory_percent": float(stats["MemPerc"].rstrip("%")),
                        "network_io": stats["NetIO"],
                        "block_io": stats["BlockIO"],
                        "pids": stats["PIDs"],
                    }
                )
                file.flush()
                time.sleep(args.interval)

        except KeyboardInterrupt:
            print("\nMonitoring stopped.", flush=True)


if __name__ == "__main__":
    main()
