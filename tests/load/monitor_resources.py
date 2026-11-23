"""Sample Docker or Podman container resource usage during load testing.

Docker reports the CPU percentage of the last interval directly. With
``--no-stream``, Podman reports the average since the container started instead,
so for Podman the CPU percentage is computed from the difference of the
cumulative CPU time (``CPUNano``) between consecutive samples.

The monitor stops cleanly with Ctrl+C (SIGINT) or SIGTERM, so it can also be
stopped from scripts with ``kill``.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import subprocess
import time

FIELDS = [
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


def run_stats(engine, container):
    """Return the first JSON object printed by ``<engine> stats``."""
    result = subprocess.run(
        [engine, "stats", "--no-stream", "--format", "{{json .}}", container],
        capture_output=True,
        text=True,
        check=True,
    )
    if not result.stdout.strip():
        raise RuntimeError(f"{engine} returned no container metrics.")
    return json.loads(result.stdout.strip().splitlines()[0])


def docker_row(stats):
    """Convert one ``docker stats`` sample to a CSV row."""
    return {
        "cpu_percent": float(stats["CPUPerc"].rstrip("%")),
        "memory_usage": stats["MemUsage"],
        "memory_limit": stats["MemUsage"].split(" / ")[-1],
        "memory_percent": float(stats["MemPerc"].rstrip("%")),
        "network_io": stats["NetIO"],
        "block_io": stats["BlockIO"],
        "pids": stats["PIDs"],
    }


def format_bytes(value):
    """Format bytes with binary units, as ``docker stats`` does (e.g. 185.3MiB)."""
    for unit, size in (("GiB", 1024**3), ("MiB", 1024**2), ("KiB", 1024)):
        if value >= size:
            return f"{value / size:.2f}{unit}"
    return f"{value}B"


def podman_cpu_percent(previous, current):
    """CPU percentage between two Podman samples (100% = one full core)."""
    elapsed = current["SystemNano"] - previous["SystemNano"]
    if elapsed <= 0:
        return 0.0
    return (current["CPUNano"] - previous["CPUNano"]) / elapsed * 100


def podman_row(previous, current):
    """Convert two consecutive ``podman stats`` samples to a CSV row."""
    limit = format_bytes(current["MemLimit"])
    return {
        "cpu_percent": round(podman_cpu_percent(previous, current), 2),
        "memory_usage": f"{format_bytes(current['MemUsage'])} / {limit}",
        "memory_limit": limit,
        "memory_percent": round(current["MemPerc"], 2),
        "network_io": (
            f"{format_bytes(current['NetInput'])} / {format_bytes(current['NetOutput'])}"
        ),
        "block_io": (
            f"{format_bytes(current['BlockInput'])} / {format_bytes(current['BlockOutput'])}"
        ),
        "pids": current["PIDs"],
    }


def stop_on_sigterm(signum, frame):
    raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser(
        description="Monitor CPU and memory usage of a Docker or Podman container."
    )
    parser.add_argument(
        "--container",
        default="mibici-api-baseline",
        help="Container name or ID (with Compose: $(docker compose ps -q api)).",
    )
    parser.add_argument(
        "--engine",
        choices=["docker", "podman"],
        default="docker",
        help="Container engine used to read the statistics.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Pause between samples in seconds (each sample also takes about 1 s).",
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

    signal.signal(signal.SIGTERM, stop_on_sigterm)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        file.flush()

        print(
            f"Monitoring {args.container} with {args.engine} every {args.interval}s. "
            "Press Ctrl+C to stop.",
            flush=True,
        )

        try:
            previous = None
            while True:
                current = run_stats(args.engine, args.container)
                if args.engine == "docker":
                    row = docker_row(current)
                elif previous is not None:
                    row = podman_row(previous, current)
                else:
                    # Podman needs two samples to compute the CPU of an interval.
                    row = None
                previous = current

                if row is not None:
                    writer.writerow(
                        {
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            "container": args.container,
                            **row,
                        }
                    )
                    file.flush()
                time.sleep(args.interval)

        except KeyboardInterrupt:
            print("\nMonitoring stopped.", flush=True)


if __name__ == "__main__":
    main()
