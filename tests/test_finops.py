"""Pruebas sintéticas del análisis FinOps."""

import json

import pandas as pd
import pytest

from mibici.finops import (
    BASELINE_RUN,
    CAPACITY_RUNS,
    OPTIMIZED_RUN,
    DemandScenario,
    DeploymentOption,
    analyze,
    batch_cost,
    capacity_steps,
    cost_per_million,
    demand_profile,
    demand_scenarios,
    fargate_hourly,
    parse_memory_mib,
    render_markdown,
    scenario_cost,
    sustainable_capacity,
    write_outputs,
)

PRICING = {
    "region": "mx-central-1",
    "region_name": "Mexico (Central)",
    "retrieved_at": "2026-11-22",
    "hours_per_month": 730,
    "sources": {},
    "ec2": {
        "t3.small": {"usd_per_hour": 0.02},
        "t3.medium": {"usd_per_hour": 0.04},
        "t4g.small": {"usd_per_hour": 0.016},
    },
    "fargate": {"x86_64": {"usd_per_vcpu_hour": 0.04, "usd_per_gb_hour": 0.005}},
    "us_east_1_comparison": {"ec2_usd_per_hour": {"t3.small": 0.019}},
}


def make_history(steps, seconds_per_step=20, start=1_700_000_000):
    """Historial de Locust: (usuarios, req/s, p95, fallos/s) constantes por escalón."""
    rows = [{"Timestamp": start, "User Count": 0, "Name": "Aggregated", "Requests/s": 0}]
    timestamp = start + 1
    for users, rps, p95, failures in steps:
        for _ in range(seconds_per_step):
            rows.append(
                {
                    "Timestamp": timestamp,
                    "User Count": users,
                    "Type": "",
                    "Name": "Aggregated",
                    "Requests/s": rps,
                    "Failures/s": failures,
                    "50%": p95 / 2,
                    "95%": p95,
                }
            )
            timestamp += 1
    return pd.DataFrame(rows)


def make_resources(cpu, memory="200MiB / 512MiB", samples=5):
    return pd.DataFrame(
        {
            "timestamp_utc": ["2026-11-22T00:00:00+00:00"] * samples,
            "cpu_percent": [cpu] * samples,
            "memory_usage": [memory] * samples,
            "memory_limit": [memory.split(" / ")[1]] * samples,
        }
    )


def make_profile():
    return {
        "stations": 120,
        "trips_total": 876_000,
        "trips_per_day_mean": 2400.0,
        "peak_trips_per_hour": 360,
        "operating_hours_per_day": 20,
        "closed_hours": [1, 2, 3, 4],
    }


def test_parse_memory_mib_handles_docker_and_podman_units():
    assert parse_memory_mib("185.3MiB / 7.6GiB") == pytest.approx(185.3)
    assert parse_memory_mib("1.5GiB / 2GiB") == pytest.approx(1536)
    assert parse_memory_mib("512KiB / 1GiB") == pytest.approx(0.5)
    assert parse_memory_mib("100MB / 1GB") == pytest.approx(100 * 1000**2 / 1024**2)
    with pytest.raises(ValueError, match="no soportada"):
        parse_memory_mib("12 parsecs")


def test_capacity_steps_skips_warmup_and_ignores_idle_rows():
    history = make_history([(1, 50, 20, 0), (4, 80, 120, 0)])
    # El arranque de un escalón mezcla valores del anterior: deben descartarse.
    history.loc[history.index[1:4], "Requests/s"] = 999

    steps = capacity_steps(history, warmup_seconds=5)

    assert steps["users"].tolist() == [1, 4]
    assert steps["rps"].tolist() == [50, 80]
    assert steps["p95_ms"].tolist() == [20, 120]
    assert steps["seconds"].tolist() == [15, 15]


def test_capacity_steps_requires_loaded_steps():
    with pytest.raises(ValueError, match="escalones"):
        capacity_steps(make_history([]))


def test_sustainable_capacity_picks_fastest_step_within_slo():
    steps = capacity_steps(
        make_history([(1, 50, 20, 0), (4, 90, 250, 0), (8, 95, 600, 0), (16, 99, 100, 5)]),
        warmup_seconds=2,
    )

    best = sustainable_capacity(steps, slo_p95_ms=300, max_failure_rate=0.01)

    # 8 usuarios supera el SLO de latencia y 16 el de errores.
    assert best["users"] == 4
    assert best["rps"] == 90
    with pytest.raises(ValueError, match="Ningún"):
        sustainable_capacity(steps, slo_p95_ms=10)


def test_demand_profile_detects_closed_hours():
    hours = pd.date_range("2025-01-01", periods=48, freq="h")
    departures = [0 if hour.hour in (2, 3) else 5 for hour in hours]
    hourly = pd.DataFrame(
        {
            "station_id": [1, 2] * 24,
            "hour": hours,
            "departures": departures,
        }
    ).query("departures > 0")

    profile = demand_profile(hourly)

    assert profile["stations"] == 2
    assert profile["closed_hours"] == [2, 3]
    assert profile["operating_hours_per_day"] == 22
    assert profile["trips_total"] == 5 * 44
    assert profile["peak_trips_per_hour"] == 5
    assert profile["days"] == 2


def test_demand_scenarios_scale_with_queries_and_growth():
    batch, app, growth = demand_scenarios(
        make_profile(), hours_per_month=720, queries_per_trip=3, growth_factor=10
    )

    assert batch.monthly_inferences == 120 * 20 * 30
    assert batch.peak_rps == 2
    assert app.monthly_inferences == 2400 * 3 * 30
    assert app.peak_rps == pytest.approx(360 * 3 / 3600)
    assert growth.monthly_inferences == pytest.approx(app.monthly_inferences * 10)


def test_costs_and_replicas():
    assert cost_per_million(0.036, 10) == pytest.approx(1.0)
    assert fargate_hourly(PRICING, 1, 2) == pytest.approx(0.05)

    option = DeploymentOption("x", "X", BASELINE_RUN, 0.02, 730, "")
    small = DemandScenario("s", "S", 1_000_000, 5.0, "")
    large = DemandScenario("l", "L", 10_000_000, 50.0, "")

    cost = scenario_cost(option, capacity_rps=10, scenario=small)
    assert cost["replicas"] == 1
    assert cost["monthly_usd"] == pytest.approx(0.02 * 730)
    assert cost["usd_per_million"] == pytest.approx(0.02 * 730)
    # 50 req/s con 7 req/s útiles por réplica (70 % de 10) requieren 8 réplicas.
    assert scenario_cost(option, capacity_rps=10, scenario=large)["replicas"] == 8


def test_batch_cost_bills_at_least_one_minute():
    short = batch_cost(PRICING, stations=100, runs_per_month=600, capacity_rps=100)
    assert short["seconds_per_run"] == pytest.approx(31)
    assert short["billed_seconds_per_run"] == 60
    assert short["monthly_usd"] == pytest.approx(600 * 60 / 3600 * 0.05)

    long = batch_cost(PRICING, stations=1000, runs_per_month=600, capacity_rps=10)
    assert long["billed_seconds_per_run"] == pytest.approx(130)


@pytest.fixture
def capacity_dir(tmp_path):
    for run in CAPACITY_RUNS:
        rps = 90 if run.name == OPTIMIZED_RUN else 60
        steps = [(1, 50, 20, 0), (16, 25, 900, 0)]
        if run.name == OPTIMIZED_RUN:
            steps = [(1, 15, 70, 0), (16, rps, 250, 0)]
        make_history(steps).to_csv(tmp_path / f"{run.name}_stats_history.csv", index=False)
        make_resources(95).to_csv(tmp_path / f"{run.name}_api_resources.csv", index=False)
        make_resources(2, "1.7GiB / 2GiB").to_csv(
            tmp_path / f"{run.name}_mlflow_resources.csv", index=False
        )
    return tmp_path


def test_analyze_and_outputs(capacity_dir, tmp_path):
    summary = analyze(capacity_dir, PRICING, make_profile(), slo_p95_ms=300)

    options = {option["key"]: option for option in summary["options"]}
    assert options["compose_t3_medium"]["capacity_rps"] == 50
    assert options["api_t3_small"]["capacity_rps"] == 90
    assert options["api_t4g_small_scheduled"]["hours_per_month"] == pytest.approx(730 * 20 / 24)

    optimizations = summary["optimizations"]
    assert optimizations["separate_mlflow"] == pytest.approx(0.5)
    assert optimizations["graviton"] == pytest.approx(0.2)
    assert optimizations["schedule_off_hours"] == pytest.approx(4 / 24)
    assert optimizations["process_concurrency"]["baseline_rps"] == 25
    assert optimizations["process_concurrency"]["optimized_rps"] == 90
    assert len(summary["scenario_costs"]) == len(summary["options"]) * 3

    markdown = render_markdown(summary)
    assert "Hallazgo principal" in markdown
    assert "mx-central-1" in markdown

    output = tmp_path / "out"
    write_outputs(summary, output)
    assert json.loads((output / "finops_summary.json").read_text(encoding="utf-8"))["options"]
    assert (output / "capacity_by_configuration.png").stat().st_size > 0
    assert (output / "cost_by_deployment_option.png").stat().st_size > 0
