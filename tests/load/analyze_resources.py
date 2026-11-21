
"""Analyze Docker resource usage within a Locust benchmark window."""

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def memory_to_mib(value):
    """Convert Docker memory usage to MiB."""
    usage = value.split("/")[0].strip()

    units = {
        "GiB": 1024,
        "MiB": 1,
        "KiB": 1 / 1024,
        "GB": 1000 / 1024,
        "MB": 1000**2 / 1024**2,
        "kB": 1000 / 1024**2,
        "B": 1 / 1024**2,
    }

    for unit in sorted(units, key=len, reverse=True):
        if usage.endswith(unit):
            return float(usage.removesuffix(unit).strip()) * units[unit]

    raise ValueError(f"Unsupported memory unit: {usage}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", required=True)
    parser.add_argument("--resources", required=True)
    args = parser.parse_args()

    history = read_csv(args.history)
    resources = read_csv(args.resources)

    if not history or not resources:
        raise SystemExit("One or both CSV files contain no observations.")

    # Use the actual Locust history timestamps.
    timestamps = [
        int(row["Timestamp"])
        for row in history
        if row["Name"] == "Aggregated"
    ]

    start = min(timestamps)
    end = max(timestamps)

    # Include the final one-second history interval.
    end += 1

    samples = []

    for row in resources:
        timestamp = datetime.fromisoformat(
            row["timestamp_utc"]
        ).timestamp()

        if start <= timestamp < end:
            samples.append(row)

    if not samples:
        raise SystemExit("No resource samples overlap the Locust window.")

    cpu = [float(row["cpu_percent"]) for row in samples]
    memory = [memory_to_mib(row["memory_usage"]) for row in samples]

    print("Benchmark window:")
    print(
        "  Start:",
        datetime.fromtimestamp(start, timezone.utc).isoformat(),
    )
    print(
        "  End:",
        datetime.fromtimestamp(end, timezone.utc).isoformat(),
    )

    print(f"\nResource samples within window: {len(samples)}")

    print("\nCPU usage:")
    print(f"  Average: {mean(cpu):.2f}%")
    print(f"  Minimum: {min(cpu):.2f}%")
    print(f"  Maximum: {max(cpu):.2f}%")

    print("\nMemory usage:")
    print(f"  Average: {mean(memory):.2f} MiB")
    print(f"  Minimum: {min(memory):.2f} MiB")
    print(f"  Maximum: {max(memory):.2f} MiB")


if __name__ == "__main__":
    main()
