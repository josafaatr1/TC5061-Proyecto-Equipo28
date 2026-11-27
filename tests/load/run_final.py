"""Run the four P2 load scenarios and export results from Locust's final JSON."""

import argparse
import csv
from datetime import UTC, datetime
import json
from pathlib import Path
import re
from statistics import mean
import subprocess
import sys

from analyze_resources import memory_to_mib
from locust.stats import calculate_response_time_percentile

SUITE = Path(__file__).resolve().parent
PREDICT = re.compile(r'"POST /predict HTTP/[^"]+" (\d{3})')


def utc(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, UTC).isoformat()


def scenario(out: Path, host: str, container: str, users: int) -> dict:
    prefix = out / f"u{users}"
    resources = prefix.with_suffix(".resources.csv")
    start = datetime.now(UTC).isoformat()
    args = [
        sys.executable,
        "-m",
        "locust",
        "-f",
        str(SUITE / "locustfile.py"),
        "--headless",
        "--host",
        host,
        "-u",
        str(users),
        "-r",
        str(users),
        "-t",
        "60s",
        "--only-summary",
        "--json-file",
        str(prefix),
        "--exit-code-on-error",
        "1",
        "--stop-timeout",
        "5",
    ]
    print(f"{start}: {users} users", flush=True)
    with (
        prefix.with_suffix(".locust.log").open("w", encoding="utf-8") as log,
        prefix.with_suffix(".monitor.log").open("w", encoding="utf-8") as monitor_log,
        subprocess.Popen(
            [
                sys.executable,
                str(SUITE / "monitor_resources.py"),
                "--engine",
                "docker",
                "--container",
                container,
                "--interval",
                "1",
                "--output",
                str(resources),
            ],
            stdout=monitor_log,
            stderr=subprocess.STDOUT,
        ) as monitor,
    ):
        try:
            code = subprocess.run(
                args, check=False, stdout=log, stderr=subprocess.STDOUT, timeout=180
            ).returncode
            end = datetime.now(UTC).isoformat()
            if monitor.poll() is not None:
                raise RuntimeError(f"Resource monitor exited early; see {prefix}.monitor.log")
        finally:
            if monitor.poll() is None:
                monitor.terminate()
    if code != 0:
        raise RuntimeError(f"Locust exited with {code}; see {prefix}.locust.log")

    # --json-file expects a prefix and appends .json; --json must not be combined with it.
    rows = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    if len(rows) != 1 or rows[0]["name"] != "POST /predict" or rows[0]["method"] != "POST":
        raise ValueError(f"Expected exactly one POST /predict row in {prefix}.json")
    row = rows[0]
    counts = {int(ms): n for ms, n in row["response_times"].items()}
    requests = row["num_requests"]
    failures = row["num_failures"]
    if (
        not requests
        or failures
        or row["num_none_requests"]
        or sum(counts.values()) != requests
        or sum(row["num_reqs_per_sec"].values()) != requests
    ):
        raise ValueError(f"Invalid request counts or failures in {prefix}.json")

    logs = subprocess.run(
        ["docker", "logs", "--since", start, "--until", end, container],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    statuses = PREDICT.findall(logs.stdout + logs.stderr)
    if statuses.count("200") != requests or len(statuses) != requests:
        raise ValueError(f"Locust and API access-log counts differ for {users} users")

    first, last = row["start_time"], row["last_request_timestamp"]
    with resources.open(encoding="utf-8", newline="") as stream:
        samples = [
            item
            for item in csv.DictReader(stream)
            if first <= datetime.fromisoformat(item["timestamp_utc"]).timestamp() <= last
        ]
    if not samples:
        raise ValueError(f"No resource samples in the Locust request window for {users} users")
    cpu = [float(item["cpu_percent"]) for item in samples]
    memory = [memory_to_mib(item["memory_usage"]) for item in samples]
    return {
        "users": users,
        "spawn_rate": users,
        "configured_seconds": 60,
        "exit_code": code,
        "requests": requests,
        "http_200": len(statuses),
        "failures": failures,
        "requests_per_second": round(requests / (last - first), 3),
        "mean_ms": round(row["total_response_time"] / requests, 3),
        "p95_ms": calculate_response_time_percentile(counts, requests, 0.95),
        "p99_ms": calculate_response_time_percentile(counts, requests, 0.99),
        "mean_cpu_pct": round(mean(cpu), 2),
        "peak_cpu_pct": max(cpu),
        "mean_ram_mib": round(mean(memory), 2),
        "peak_ram_mib": max(memory),
        "resource_samples": len(samples),
        "start_utc": utc(first),
        "last_response_utc": utc(last),
    }


def plot(rows: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(9, 6), layout="constrained")
    users = [row["users"] for row in rows]
    for axis, title, series in zip(
        axes.flat,
        ("Latencia (ms)", "Peticiones por segundo", "CPU de API (%)", "Memoria de API (MiB)"),
        (
            {"mean_ms": "Media", "p95_ms": "P95", "p99_ms": "P99"},
            {"requests_per_second": "RPS"},
            {"mean_cpu_pct": "Media", "peak_cpu_pct": "Pico"},
            {"mean_ram_mib": "Media", "peak_ram_mib": "Pico"},
        ),
        strict=True,
    ):
        for key, label in series.items():
            axis.plot(users, [row[key] for row in rows], marker="o", label=label)
        upper = max(row[key] for row in rows for key in series)
        axis.set(title=title, xlabel="Usuarios", xticks=users, ylim=(0, max(upper * 1.12, 1)))
        axis.grid(axis="y", alpha=0.25)
        if len(series) > 1:
            axis.legend()
    fig.suptitle("MiBici P2: carga con espera de 1 a 2 s")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--container", required=True, help="Compose API container ID")
    parser.add_argument("--output", required=True, type=Path, help="New, unused output directory")
    opts = parser.parse_args()
    output = opts.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for users in (1, 5, 10, 20):
        row = scenario(output, opts.host, opts.container, users)
        rows.append(row)
        print(json.dumps(row), flush=True)
    with (output / "load_summary.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot(rows, output / "load_summary.png")


if __name__ == "__main__":
    main()
