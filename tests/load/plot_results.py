
"""Generate reproducible MiBici API load-test graphs.

Sources:
- Locust statistics CSV files.
- Locust statistics history CSV files.
- Docker resource monitoring CSV files.

Resource measurements are filtered to the corresponding Locust
benchmark window before calculating CPU and memory statistics.
"""

import csv
import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "graphs"

SCENARIOS = [
    ("baseline_1user", 1, "Baseline"),
    ("moderate_5users", 5, "Moderate"),
    ("high_10users", 10, "High"),
    ("stress_20users", 20, "Stress"),
]


# ---------------------------------------------------------------------
# CSV utilities
# ---------------------------------------------------------------------

def read_csv(path: Path) -> list[dict]:
    """Read a CSV file and return its rows."""

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open(
        newline="",
        encoding="utf-8",
    ) as file:
        return list(csv.DictReader(file))


def parse_memory(value: str) -> float:
    """Convert Docker memory measurements to MiB."""

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
            return (
                float(usage.removesuffix(unit).strip())
                * units[unit]
            )

    raise ValueError(
        f"Unsupported memory unit: {usage}"
    )

# ---------------------------------------------------------------------
# Benchmark window alignment
# ---------------------------------------------------------------------

def get_benchmark_window(history: list[dict]) -> tuple[float, float]:
    """Extract the benchmark window from aggregated Locust records."""

    timestamps = [
        int(row["Timestamp"])
        for row in history
        if row.get("Name") == "Aggregated"
        and row.get("Timestamp")
        and row["Timestamp"] != "N/A"
    ]

    if not timestamps:
        raise ValueError(
            "No aggregated timestamps found in Locust history."
        )

    start = min(timestamps)
    end = max(timestamps) + 1

    return start, end

def filter_resources_by_benchmark(
    resources: list[dict],
    start: float,
    end: float,
) -> list[dict]:
    """Keep Docker samples within the Locust benchmark window."""

    filtered = []

    for row in resources:
        timestamp = datetime.fromisoformat(
            row["timestamp_utc"]
        ).timestamp()

        if start <= timestamp < end:
            filtered.append(row)

    if not filtered:
        raise ValueError(
            "No Docker resource samples found within "
            "the Locust benchmark window."
        )

    return filtered
# ---------------------------------------------------------------------
# Benchmark data collection
# ---------------------------------------------------------------------

def collect_results() -> list[dict]:
    """Collect HTTP and resource metrics for all scenarios."""

    results = []

    for filename, users, label in SCENARIOS:

        print(f"\nProcessing scenario: {label}")

        # Read original evidence
        stats = read_csv(
            RESULTS / f"{filename}_stats.csv"
        )

        history = read_csv(
            RESULTS / f"{filename}_stats_history.csv"
        )

        resources = read_csv(
            RESULTS / f"{filename}_resources.csv"
        )

        # Determine the actual Locust benchmark window
        start, end = get_benchmark_window(history)

        # Align Docker samples with the benchmark window
        aligned_resources = filter_resources_by_benchmark(
            resources,
            start,
            end,
        )

        # Retrieve the prediction endpoint measurements
        aggregated = next(
            (
                row for row in stats
                if row["Name"] == "POST /predict"
            ),
            None,
        )

        if aggregated is None:
            raise ValueError(
                f"POST /predict not found in {filename}_stats.csv"
            )

        # HTTP performance
        requests = int(aggregated["Request Count"])
        failures = int(aggregated["Failure Count"])

        throughput = float(aggregated["Requests/s"])

        avg_latency = float(
            aggregated["Average Response Time"]
        )

        p50 = float(aggregated["50%"])
        p95 = float(aggregated["95%"])
        p99 = float(aggregated["99%"])

        # CPU measurements within benchmark window
        cpu = [
            float(row["cpu_percent"])
            for row in aligned_resources
        ]

        # Memory measurements within benchmark window
        memory = [
            parse_memory(row["memory_usage"])
            for row in aligned_resources
        ]

        # Resource monitoring window
        start_iso = datetime.fromtimestamp(
            start,
            tz=timezone.utc,
        ).isoformat()

        end_iso = datetime.fromtimestamp(
            end,
            tz=timezone.utc,
        ).isoformat()

        result = {
            "scenario": label,
            "users": users,

            # HTTP metrics
            "requests": requests,
            "failures": failures,
            "failure_rate_percent": (
                failures / requests * 100
                if requests else 0
            ),
            "throughput": throughput,
            "avg_latency": avg_latency,
            "p50": p50,
            "p95": p95,
            "p99": p99,

            # CPU metrics
            "cpu_avg": sum(cpu) / len(cpu),
            "cpu_min": min(cpu),
            "cpu_max": max(cpu),

            # Memory metrics
            "memory_avg": sum(memory) / len(memory),
            "memory_min": min(memory),
            "memory_max": max(memory),

            # Evidence metadata
            "resource_samples": len(aligned_resources),
            "resource_window_start": start_iso,
            "resource_window_end": end_iso,
        }

        results.append(result)

        print(
            f"  Benchmark window: {start_iso} - {end_iso}"
        )

        print(
            f"  Resource samples: {len(aligned_resources)}"
        )

        print(
            f"  CPU average: {result['cpu_avg']:.2f}%"
        )

        print(
            f"  Memory average: "
            f"{result['memory_avg']:.2f} MiB"
        )

    return results


# ---------------------------------------------------------------------
# Graph utilities
# ---------------------------------------------------------------------

def save_graph(filename: str) -> None:
    """Save a graph as a high-resolution PNG."""

    plt.tight_layout()

    plt.savefig(
        OUTPUT / filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Generated: {filename}")


# ---------------------------------------------------------------------
# Figure 1: Response latency
# ---------------------------------------------------------------------

def plot_latency(data: list[dict]) -> None:

    users = [row["users"] for row in data]

    plt.figure(figsize=(10, 6))

    metrics = [
        ("avg_latency", "Average latency"),
        ("p95", "P95 latency"),
        ("p99", "P99 latency"),
    ]

    for key, label in metrics:

        values = [
            row[key]
            for row in data
        ]

        plt.plot(
            users,
            values,
            marker="o",
            linewidth=2,
            markersize=7,
            label=label,
        )

    plt.title(
        "MiBici API - Response Latency",
        fontsize=14,
        fontweight="bold",
    )

    plt.xlabel("Concurrent users")
    plt.ylabel("Response time (ms)")

    plt.xticks(users)

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    save_graph("01_response_latency.png")


# ---------------------------------------------------------------------
# Figure 2: Throughput
# ---------------------------------------------------------------------

def plot_throughput(data: list[dict]) -> None:

    users = [row["users"] for row in data]

    throughput = [
        row["throughput"]
        for row in data
    ]

    plt.figure(figsize=(10, 6))

    bars = plt.bar(
        [str(user) for user in users],
        throughput,
        width=0.6,
    )

    for bar, value in zip(bars, throughput):

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
        )

    plt.title(
        "MiBici API - Throughput",
        fontsize=14,
        fontweight="bold",
    )

    plt.xlabel("Concurrent users")
    plt.ylabel("Requests per second")

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    save_graph("02_throughput.png")


# ---------------------------------------------------------------------
# Figure 3: CPU utilization
# ---------------------------------------------------------------------

def plot_cpu(data: list[dict]) -> None:

    users = [row["users"] for row in data]

    plt.figure(figsize=(10, 6))

    plt.plot(
        users,
        [row["cpu_avg"] for row in data],
        marker="o",
        linewidth=2,
        label="Average CPU",
    )

    plt.plot(
        users,
        [row["cpu_max"] for row in data],
        marker="s",
        linewidth=2,
        label="Maximum CPU",
    )

    plt.title(
        "MiBici API - CPU Utilization",
        fontsize=14,
        fontweight="bold",
    )

    plt.xlabel("Concurrent users")
    plt.ylabel("CPU utilization (%)")

    plt.xticks(users)

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    save_graph("03_cpu_utilization.png")


# ---------------------------------------------------------------------
# Figure 4: Memory utilization
# ---------------------------------------------------------------------

def plot_memory(data: list[dict]) -> None:

    users = [row["users"] for row in data]

    plt.figure(figsize=(10, 6))

    plt.plot(
        users,
        [row["memory_avg"] for row in data],
        marker="o",
        linewidth=2,
        label="Average memory",
    )

    plt.plot(
        users,
        [row["memory_max"] for row in data],
        marker="s",
        linewidth=2,
        label="Maximum memory",
    )

    plt.title(
        "MiBici API - Memory Utilization",
        fontsize=14,
        fontweight="bold",
    )

    plt.xlabel("Concurrent users")
    plt.ylabel("Memory usage (MiB)")

    plt.xticks(users)

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    save_graph("04_memory_utilization.png")


# ---------------------------------------------------------------------
# Consolidated benchmark report
# ---------------------------------------------------------------------

def export_summary(data: list[dict]) -> None:
    """Export all measured metrics to a consolidated CSV."""

    output_file = OUTPUT / "benchmark_summary.csv"

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        fields = list(data[0].keys())

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(data)

    print(f"\nSummary exported: {output_file}")


def print_summary(data: list[dict]) -> None:
    """Print a comparative benchmark summary."""

    print("\n" + "=" * 100)
    print("MIBICI API - BENCHMARK SUMMARY")
    print("=" * 100)

    print(
        f"{'Scenario':<12}"
        f"{'Users':>7}"
        f"{'Requests':>11}"
        f"{'Failures':>11}"
        f"{'Req/s':>10}"
        f"{'Avg ms':>11}"
        f"{'P95 ms':>10}"
        f"{'P99 ms':>10}"
        f"{'CPU %':>10}"
        f"{'RAM MiB':>11}"
    )

    print("-" * 100)

    for row in data:

        print(
            f"{row['scenario']:<12}"
            f"{row['users']:>7}"
            f"{row['requests']:>11}"
            f"{row['failures']:>11}"
            f"{row['throughput']:>10.2f}"
            f"{row['avg_latency']:>11.2f}"
            f"{row['p95']:>10.2f}"
            f"{row['p99']:>10.2f}"
            f"{row['cpu_avg']:>10.2f}"
            f"{row['memory_avg']:>11.2f}"
        )

    print("-" * 100)

    print("\nRESOURCE WINDOW VALIDATION")

    for row in data:

        print(
            f"{row['scenario']:<12} "
            f"| Samples: {row['resource_samples']:>3} "
            f"| CPU min/avg/max: "
            f"{row['cpu_min']:.2f}/"
            f"{row['cpu_avg']:.2f}/"
            f"{row['cpu_max']:.2f}% "
            f"| Memory min/avg/max: "
            f"{row['memory_min']:.2f}/"
            f"{row['memory_avg']:.2f}/"
            f"{row['memory_max']:.2f} MiB"
        )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Starting MiBici benchmark reporting...")

    data = collect_results()

    print("\nGenerating figures...")

    plot_latency(data)
    plot_throughput(data)
    plot_cpu(data)
    plot_memory(data)

    export_summary(data)

    print_summary(data)

    print("\nReporting completed successfully.")
    print(f"Output directory: {OUTPUT}")


if __name__ == "__main__":
    main()
