"""Análisis FinOps: costo por inferencia de la API de predicción P2.

Combina tres fuentes:

- Capacidad medida: pruebas de capacidad de Locust (``tests/load/locustfile_capacity.py``)
  contra la API de Docker Compose con distintos límites de CPU y procesos, más el
  monitoreo de ``docker stats`` de la API y de MLflow.
- Precios de referencia: precios on-demand oficiales de AWS en ``references/``.
- Demanda real: viajes por hora de MiBici en 2025 (``station_hourly_2025.parquet``).

Con ellos estima el costo por millón de inferencias a capacidad sostenible, el costo
mensual y por inferencia para escenarios de demanda realistas y el ahorro de cada
optimización.

    python -m mibici.finops
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

from loguru import logger
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import typer

from mibici.config import (
    PROCESSED_DATA_DIR,
    PROJ_ROOT,
    REFERENCES_DIR,
    REPORTS_DIR,
)

app = typer.Typer()

CAPACITY_DIR = PROJ_ROOT / "tests" / "load" / "results" / "capacity"
PRICING_PATH = REFERENCES_DIR / "aws_pricing_mx-central-1.json"
HOURLY_DATA_PATH = PROCESSED_DATA_DIR / "station_hourly_2025.parquet"
OUTPUT_DIR = REPORTS_DIR / "finops"

# Latencia máxima aceptable (p95) para una consulta interactiva y holgura de capacidad:
# se planea usar como máximo el 70 % de la capacidad sostenible para absorber picos.
SLO_P95_MS = 300.0
MAX_FAILURE_RATE = 0.01
TARGET_UTILIZATION = 0.7
# Consultas a la API por viaje en el escenario de app: disponibilidad en el origen,
# anclajes en el destino y una actualización.
QUERIES_PER_TRIP = 3
GROWTH_FACTOR = 10
# Segundos de arranque de un contenedor en un lote programado (carga de los modelos).
BATCH_STARTUP_SECONDS = 30.0
WARMUP_SECONDS = 10


@dataclass(frozen=True)
class CapacityRun:
    """Configuración de la API durante una prueba de capacidad."""

    name: str
    label: str
    vcpu: float
    memory_limit_gib: float
    workers: int
    omp_threads: int | None


CAPACITY_RUNS = (
    CapacityRun("capacity", "1 vCPU, 1 proceso", 1.0, 0.5, 1, None),
    CapacityRun("capacity_omp1", "1 vCPU, 1 proceso, OMP=1", 1.0, 0.5, 1, 1),
    CapacityRun("capacity_2cpu", "2 vCPU, 1 proceso", 2.0, 0.5, 1, None),
    CapacityRun("capacity_2cpu_2workers", "2 vCPU, 2 procesos", 2.0, 1.0, 2, None),
    CapacityRun("capacity_2cpu_2workers_omp1", "2 vCPU, 2 procesos, OMP=1", 2.0, 1.0, 2, 1),
)
BASELINE_RUN = "capacity"
OPTIMIZED_RUN = "capacity_2cpu_2workers_omp1"


# ---------------------------------------------------------------------------
# Lectura de mediciones
# ---------------------------------------------------------------------------

MEMORY_UNITS = {
    "GiB": 1024,
    "MiB": 1,
    "KiB": 1 / 1024,
    "GB": 1000**3 / 1024**2,
    "MB": 1000**2 / 1024**2,
    "kB": 1000 / 1024**2,
    "B": 1 / 1024**2,
}


def parse_memory_mib(value: str) -> float:
    """Convierte el uso de memoria de ``docker stats`` (``"185.3MiB / 512MiB"``) a MiB."""
    usage = value.split("/")[0].strip()
    for unit in sorted(MEMORY_UNITS, key=len, reverse=True):
        if usage.endswith(unit):
            return float(usage.removesuffix(unit).strip()) * MEMORY_UNITS[unit]
    raise ValueError(f"Unidad de memoria no soportada: {usage}")


def capacity_steps(history: pd.DataFrame, warmup_seconds: int = WARMUP_SECONDS) -> pd.DataFrame:
    """Resume cada escalón de usuarios del historial de Locust.

    Descarta los primeros ``warmup_seconds`` de cada escalón, en los que las ventanas
    móviles de Locust aún mezclan el escalón anterior, y usa la mediana del resto.
    """
    aggregated = history[(history["Name"] == "Aggregated") & (history["User Count"] > 0)]
    rows = []
    for users, group in aggregated.sort_values("Timestamp").groupby("User Count"):
        stable = group.iloc[warmup_seconds:]
        if stable.empty:
            continue
        rps = stable["Requests/s"].median()
        failures = stable["Failures/s"].median()
        rows.append(
            {
                "users": int(users),
                "rps": float(rps),
                "p50_ms": float(stable["50%"].median()),
                "p95_ms": float(stable["95%"].median()),
                "failure_rate": float(failures / rps) if rps else 0.0,
                "seconds": len(stable),
            }
        )
    if not rows:
        raise ValueError("El historial de Locust no contiene escalones con usuarios.")
    return pd.DataFrame(rows)


def sustainable_capacity(
    steps: pd.DataFrame,
    slo_p95_ms: float = SLO_P95_MS,
    max_failure_rate: float = MAX_FAILURE_RATE,
) -> dict:
    """Escalón con más peticiones por segundo que cumple el SLO de latencia y errores."""
    compliant = steps[
        (steps["p95_ms"] <= slo_p95_ms) & (steps["failure_rate"] <= max_failure_rate)
    ]
    if compliant.empty:
        raise ValueError(f"Ningún escalón cumple p95 <= {slo_p95_ms} ms.")
    best = compliant.loc[compliant["rps"].idxmax()]
    return {key: (int(v) if key == "users" else float(v)) for key, v in best.items()}


def resource_summary(resources: pd.DataFrame) -> dict:
    """CPU (100 % = un vCPU) y memoria de un contenedor durante la prueba."""
    memory = resources["memory_usage"].map(parse_memory_mib)
    return {
        "cpu_avg_percent": float(resources["cpu_percent"].mean()),
        "cpu_p95_percent": float(resources["cpu_percent"].quantile(0.95)),
        "memory_avg_mib": float(memory.mean()),
        "memory_max_mib": float(memory.max()),
        "samples": len(resources),
    }


def load_capacity_runs(capacity_dir: Path, slo_p95_ms: float) -> list[dict]:
    """Lee las pruebas de capacidad y calcula su capacidad sostenible y recursos."""
    results = []
    for run in CAPACITY_RUNS:
        history = pd.read_csv(capacity_dir / f"{run.name}_stats_history.csv")
        steps = capacity_steps(history)
        results.append(
            {
                **asdict(run),
                "steps": steps.to_dict(orient="records"),
                "sustainable": sustainable_capacity(steps, slo_p95_ms),
                "api": resource_summary(
                    pd.read_csv(capacity_dir / f"{run.name}_api_resources.csv")
                ),
                "mlflow": resource_summary(
                    pd.read_csv(capacity_dir / f"{run.name}_mlflow_resources.csv")
                ),
            }
        )
    return results


# ---------------------------------------------------------------------------
# Demanda real de MiBici
# ---------------------------------------------------------------------------


def demand_profile(hourly: pd.DataFrame) -> dict:
    """Perfil de demanda a partir de los viajes por estación y hora."""
    hours = pd.to_datetime(hourly["hour"])
    trips = hourly.groupby(hours)["departures"].sum()
    full_range = pd.date_range(trips.index.min(), trips.index.max(), freq="h")
    trips = trips.reindex(full_range, fill_value=0)

    mean_by_hour = trips.groupby(trips.index.hour).mean()
    closed_hours = [int(hour) for hour, mean in mean_by_hour.items() if mean < 1]
    daily = trips.resample("D").sum()
    return {
        "stations": int(hourly["station_id"].nunique()),
        "trips_total": int(trips.sum()),
        "days": len(daily),
        "trips_per_day_mean": float(daily.mean()),
        "trips_per_day_max": int(daily.max()),
        "peak_trips_per_hour": int(trips.max()),
        "closed_hours": closed_hours,
        "operating_hours_per_day": 24 - len(closed_hours),
        "mean_trips_by_hour": {int(h): round(float(v), 1) for h, v in mean_by_hour.items()},
    }


@dataclass(frozen=True)
class DemandScenario:
    key: str
    label: str
    monthly_inferences: float
    peak_rps: float
    description: str


def demand_scenarios(
    profile: dict,
    hours_per_month: float,
    queries_per_trip: int = QUERIES_PER_TRIP,
    growth_factor: int = GROWTH_FACTOR,
) -> list[DemandScenario]:
    """Escenarios de demanda derivados de los datos reales."""
    days_per_month = hours_per_month / 24
    operating_hours = profile["operating_hours_per_day"]
    stations = profile["stations"]
    app_monthly = profile["trips_per_day_mean"] * queries_per_trip * days_per_month
    app_peak = profile["peak_trips_per_hour"] * queries_per_trip / 3600
    return [
        DemandScenario(
            "batch",
            "Lote por estación",
            stations * operating_hours * days_per_month,
            stations / 60,
            f"Una predicción por estación cada hora de operación ({stations} estaciones, "
            f"{operating_hours} h/día).",
        ),
        DemandScenario(
            "app",
            "App de usuarios",
            app_monthly,
            app_peak,
            f"{queries_per_trip} consultas por viaje; pico de "
            f"{profile['peak_trips_per_hour']} viajes/hora.",
        ),
        DemandScenario(
            "app_x10",
            f"App de usuarios x{growth_factor}",
            app_monthly * growth_factor,
            app_peak * growth_factor,
            f"Crecimiento de {growth_factor} veces sobre el escenario de app.",
        ),
    ]


# ---------------------------------------------------------------------------
# Costos
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeploymentOption:
    key: str
    label: str
    run_name: str
    usd_per_hour: float
    hours_per_month: float
    notes: str


def fargate_hourly(pricing: dict, vcpu: float, memory_gb: float, arch: str = "x86_64") -> float:
    rates = pricing["fargate"][arch]
    return vcpu * rates["usd_per_vcpu_hour"] + memory_gb * rates["usd_per_gb_hour"]


def deployment_options(pricing: dict, operating_hours_per_day: int) -> list[DeploymentOption]:
    """Opciones de despliegue, cada una ligada a la prueba de capacidad que la representa."""
    ec2 = pricing["ec2"]
    month = pricing["hours_per_month"]
    scheduled = operating_hours_per_day * month / 24
    return [
        DeploymentOption(
            "compose_t3_medium",
            "Compose actual: API + MLflow en t3.medium",
            BASELINE_RUN,
            ec2["t3.medium"]["usd_per_hour"],
            month,
            "Configuración por defecto de docker-compose.yml (API 1 vCPU, MLflow 1 vCPU, "
            "3 GiB en total). MLflow ocupa la mayor parte de la memoria.",
        ),
        DeploymentOption(
            "api_t3_small",
            "API sola optimizada en t3.small",
            OPTIMIZED_RUN,
            ec2["t3.small"]["usd_per_hour"],
            month,
            "MLflow fuera de la ruta de inferencia (los modelos van en la imagen); "
            "2 procesos con OMP_NUM_THREADS=1.",
        ),
        DeploymentOption(
            "api_t4g_small",
            "API sola optimizada en t4g.small (Graviton)",
            OPTIMIZED_RUN,
            ec2["t4g.small"]["usd_per_hour"],
            month,
            "Igual que t3.small en ARM; requiere construir y validar la imagen arm64.",
        ),
        DeploymentOption(
            "api_t4g_small_scheduled",
            "t4g.small apagada en horas sin servicio",
            OPTIMIZED_RUN,
            ec2["t4g.small"]["usd_per_hour"],
            scheduled,
            f"Encendida solo {operating_hours_per_day} h/día: MiBici no registra viajes "
            "en el resto.",
        ),
        DeploymentOption(
            "fargate_1vcpu",
            "Fargate 1 vCPU / 2 GB siempre encendido",
            BASELINE_RUN,
            fargate_hourly(pricing, 1, 2),
            month,
            "Sin administrar servidores; 1 vCPU exige al menos 2 GB en Fargate.",
        ),
    ]


def usable_rps(capacity_rps: float, target_utilization: float = TARGET_UTILIZATION) -> float:
    return capacity_rps * target_utilization


def cost_per_million(usd_per_hour: float, rps: float) -> float:
    """Costo de un millón de inferencias si la instancia procesa ``rps`` sin pausa."""
    return usd_per_hour / (rps * 3600) * 1_000_000


def scenario_cost(option: DeploymentOption, capacity_rps: float, scenario: DemandScenario) -> dict:
    """Réplicas, costo mensual y costo por inferencia de una opción para una demanda."""
    per_replica = usable_rps(capacity_rps)
    replicas = max(1, math.ceil(scenario.peak_rps / per_replica))
    monthly = replicas * option.usd_per_hour * option.hours_per_month
    served_capacity = replicas * capacity_rps * option.hours_per_month * 3600
    return {
        "option": option.key,
        "scenario": scenario.key,
        "replicas": replicas,
        "monthly_usd": monthly,
        "usd_per_million": monthly / scenario.monthly_inferences * 1_000_000,
        "utilization": scenario.monthly_inferences / served_capacity,
    }


def batch_cost(
    pricing: dict,
    stations: int,
    runs_per_month: float,
    capacity_rps: float,
    startup_seconds: float = BATCH_STARTUP_SECONDS,
) -> dict:
    """Lote horario en una tarea Fargate efímera: se cobra por segundo, mínimo 60 s."""
    duration = startup_seconds + stations / capacity_rps
    billed = max(60.0, duration)
    hourly = fargate_hourly(pricing, 1, 2)
    monthly = runs_per_month * billed / 3600 * hourly
    inferences = stations * runs_per_month
    return {
        "seconds_per_run": duration,
        "billed_seconds_per_run": billed,
        "runs_per_month": runs_per_month,
        "monthly_usd": monthly,
        "usd_per_million": monthly / inferences * 1_000_000,
    }


def savings(before: float, after: float) -> float:
    return (before - after) / before


# ---------------------------------------------------------------------------
# Análisis completo
# ---------------------------------------------------------------------------


def analyze(
    capacity_dir: Path,
    pricing: dict,
    profile: dict,
    slo_p95_ms: float = SLO_P95_MS,
) -> dict:
    runs = load_capacity_runs(capacity_dir, slo_p95_ms)
    capacity = {run["name"]: run["sustainable"]["rps"] for run in runs}
    month = pricing["hours_per_month"]
    scenarios = demand_scenarios(profile, month)
    options = deployment_options(pricing, profile["operating_hours_per_day"])

    option_rows = []
    for option in options:
        rps = capacity[option.run_name]
        option_rows.append(
            {
                **asdict(option),
                "capacity_rps": rps,
                "usable_rps": usable_rps(rps),
                "monthly_usd": option.usd_per_hour * option.hours_per_month,
                "usd_per_million_at_capacity": cost_per_million(option.usd_per_hour, rps),
                "usd_per_million_planned": cost_per_million(option.usd_per_hour, usable_rps(rps)),
            }
        )
    by_option = {option.key: option for option in options}
    costs = [
        scenario_cost(option, capacity[option.run_name], scenario)
        for option in options
        for scenario in scenarios
    ]
    runs_per_month = profile["operating_hours_per_day"] * month / 24
    batch = batch_cost(pricing, profile["stations"], runs_per_month, capacity[BASELINE_RUN])

    def monthly(key: str) -> float:
        option = by_option[key]
        return option.usd_per_hour * option.hours_per_month

    baseline_run = next(run for run in runs if run["name"] == BASELINE_RUN)
    optimized_run = next(run for run in runs if run["name"] == OPTIMIZED_RUN)
    concurrency = optimized_run["sustainable"]["users"]
    baseline_same_users = next(
        step for step in baseline_run["steps"] if step["users"] == concurrency
    )
    optimizations = {
        "separate_mlflow": savings(monthly("compose_t3_medium"), monthly("api_t3_small")),
        "graviton": savings(monthly("api_t3_small"), monthly("api_t4g_small")),
        "schedule_off_hours": savings(
            monthly("api_t4g_small"), monthly("api_t4g_small_scheduled")
        ),
        "combined": savings(monthly("compose_t3_medium"), monthly("api_t4g_small_scheduled")),
        "batch_vs_always_on": savings(monthly("fargate_1vcpu"), batch["monthly_usd"]),
        "process_concurrency": {
            "users": concurrency,
            "baseline_rps": baseline_same_users["rps"],
            "baseline_p95_ms": baseline_same_users["p95_ms"],
            "optimized_rps": optimized_run["sustainable"]["rps"],
            "optimized_p95_ms": optimized_run["sustainable"]["p95_ms"],
        },
        "region_mx_vs_us_east_1": {
            instance: pricing["ec2"][instance]["usd_per_hour"] / price - 1
            for instance, price in pricing["us_east_1_comparison"]["ec2_usd_per_hour"].items()
        },
    }
    return {
        "parameters": {
            "slo_p95_ms": slo_p95_ms,
            "max_failure_rate": MAX_FAILURE_RATE,
            "target_utilization": TARGET_UTILIZATION,
            "queries_per_trip": QUERIES_PER_TRIP,
            "growth_factor": GROWTH_FACTOR,
            "batch_startup_seconds": BATCH_STARTUP_SECONDS,
            "hours_per_month": month,
        },
        "pricing": {
            "region": pricing["region"],
            "region_name": pricing["region_name"],
            "retrieved_at": pricing["retrieved_at"],
            "sources": pricing["sources"],
        },
        "capacity_runs": runs,
        "demand_profile": profile,
        "scenarios": [asdict(scenario) for scenario in scenarios],
        "options": option_rows,
        "scenario_costs": costs,
        "batch": batch,
        "optimizations": optimizations,
    }


# ---------------------------------------------------------------------------
# Salidas
# ---------------------------------------------------------------------------


def money(value: float) -> str:
    return f"${value:,.4f}" if value < 1 else f"${value:,.2f}"


def percent(value: float) -> str:
    return f"{value * 100:.1f} %"


def render_markdown(summary: dict) -> str:
    params = summary["parameters"]
    pricing = summary["pricing"]
    profile = summary["demand_profile"]
    lines = [
        "# Análisis FinOps: costo por inferencia de la API MiBici",
        "",
        (
            "Generado con `python -m mibici.finops`. No editar a mano: los valores salen de las "
            "pruebas de capacidad en `tests/load/results/capacity/`, de los precios en "
            "`references/aws_pricing_mx-central-1.json` y de los viajes de MiBici de 2025."
        ),
        "",
        "## Supuestos",
        "",
        (
            f"- Precios on-demand de AWS, región {pricing['region_name']} "
            f"(`{pricing['region']}`), consultados el {pricing['retrieved_at']}; "
            f"{params['hours_per_month']} horas por mes."
        ),
        (
            f"- Capacidad sostenible: escalón con más peticiones/s con p95 <= "
            f"{params['slo_p95_ms']:.0f} ms y errores <= {params['max_failure_rate']:.0%}."
        ),
        (
            f"- Para dimensionar se usa como máximo el {params['target_utilization']:.0%} de esa "
            "capacidad."
        ),
        (
            "- Las pruebas corrieron en Docker Desktop (AMD Ryzen 7 3700U); un vCPU de AWS puede "
            "rendir distinto, así que las capacidades son una referencia, no una garantía."
        ),
        (
            "- t3 y t4g son instancias burstable: sin consumir créditos solo sostienen el 20 % "
            "de cada vCPU. Con la demanda real (utilización menor al 10 %) eso basta, pero su "
            "costo a capacidad es optimista; para carga sostenida conviene c7g.medium."
        ),
        "",
        "## Capacidad medida",
        "",
        (
            "| Configuración de la API | Capacidad sostenible | Usuarios | p95 | CPU API prom. | "
            "Memoria API máx. | Memoria MLflow máx. |"
        ),
        "|---|---|---|---|---|---|---|",
    ]
    for run in summary["capacity_runs"]:
        best = run["sustainable"]
        lines.append(
            f"| {run['label']} | {best['rps']:.1f} req/s | {best['users']} | "
            f"{best['p95_ms']:.0f} ms | {run['api']['cpu_avg_percent']:.0f} % | "
            f"{run['api']['memory_max_mib']:.0f} MiB | {run['mlflow']['memory_max_mib']:.0f} MiB |"
        )
    process = summary["optimizations"]["process_concurrency"]
    lines += [
        "",
        (
            f"Con {process['users']} usuarios concurrentes, un solo proceso con 1 vCPU atiende "
            f"{process['baseline_rps']:.1f} req/s (p95 {process['baseline_p95_ms']:.0f} ms) y la "
            f"configuración optimizada {process['optimized_rps']:.1f} req/s "
            f"(p95 {process['optimized_p95_ms']:.0f} ms)."
        ),
        "",
        "## Demanda real de MiBici (2025)",
        "",
        (
            f"- {profile['stations']} estaciones con actividad, {profile['trips_total']:,} viajes; "
            f"{profile['trips_per_day_mean']:,.0f} viajes/día en promedio."
        ),
        (
            f"- Hora pico: {profile['peak_trips_per_hour']:,} viajes. Sin viajes en las horas "
            f"{', '.join(f'{h}:00' for h in profile['closed_hours'])} "
            f"({profile['operating_hours_per_day']} h de operación al día)."
        ),
        "",
        "| Escenario | Inferencias/mes | Pico (req/s) | Descripción |",
        "|---|---|---|---|",
    ]
    for scenario in summary["scenarios"]:
        lines.append(
            f"| {scenario['label']} | {scenario['monthly_inferences']:,.0f} | "
            f"{scenario['peak_rps']:.2f} | {scenario['description']} |"
        )
    lines += [
        "",
        "## Costo por inferencia",
        "",
        (
            "| Opción | USD/hora | USD/mes | Capacidad | USD por millón (a capacidad) | "
            "USD por millón (al 70 %) |"
        ),
        "|---|---|---|---|---|---|",
    ]
    for option in summary["options"]:
        lines.append(
            f"| {option['label']} | {money(option['usd_per_hour'])} | "
            f"{money(option['monthly_usd'])} | {option['capacity_rps']:.1f} req/s | "
            f"{money(option['usd_per_million_at_capacity'])} | "
            f"{money(option['usd_per_million_planned'])} |"
        )
    labels = {option["key"]: option["label"] for option in summary["options"]}
    scenario_labels = {s["key"]: s["label"] for s in summary["scenarios"]}
    lines += [
        "",
        "### Costo con la demanda real",
        "",
        "| Opción | Escenario | Réplicas | USD/mes | USD por millón | Utilización |",
        "|---|---|---|---|---|---|",
    ]
    for cost in summary["scenario_costs"]:
        lines.append(
            f"| {labels[cost['option']]} | {scenario_labels[cost['scenario']]} | "
            f"{cost['replicas']} | {money(cost['monthly_usd'])} | "
            f"{money(cost['usd_per_million'])} | {percent(cost['utilization'])} |"
        )
    app_costs = [cost for cost in summary["scenario_costs"] if cost["scenario"] == "app"]
    cheapest = min(app_costs, key=lambda cost: cost["usd_per_million"])
    cheapest_option = next(o for o in summary["options"] if o["key"] == cheapest["option"])
    idle_factor = cheapest["usd_per_million"] / cheapest_option["usd_per_million_at_capacity"]
    lines += [
        "",
        (
            f"Hallazgo principal: con la demanda de la app, la opción más barata "
            f"({cheapest_option['label']}) usa el {percent(cheapest['utilization'])} de su "
            f"capacidad, así que cada millón de inferencias cuesta "
            f"{money(cheapest['usd_per_million'])} en lugar de "
            f"{money(cheapest_option['usd_per_million_at_capacity'])}: unas "
            f"{idle_factor:,.0f} veces más. El costo lo domina la capacidad ociosa, no el "
            "cómputo por petición; por eso las optimizaciones con más impacto reducen horas "
            "encendidas y tamaño de instancia, no la latencia."
        ),
    ]
    batch = summary["batch"]
    opt = summary["optimizations"]
    lines += [
        "",
        (
            f"Lote horario en Fargate efímero: {batch['seconds_per_run']:.0f} s por corrida "
            f"(se cobran {batch['billed_seconds_per_run']:.0f} s), "
            f"{batch['runs_per_month']:.0f} corridas al mes: {money(batch['monthly_usd'])}/mes, "
            f"{money(batch['usd_per_million'])} por millón."
        ),
        "",
        "## Optimizaciones",
        "",
        "| Optimización | Ahorro |",
        "|---|---|",
        f"| Separar MLflow de la API (t3.medium -> t3.small) | {percent(opt['separate_mlflow'])} |",
        f"| Graviton (t3.small -> t4g.small) | {percent(opt['graviton'])} |",
        f"| Apagar en horas sin servicio | {percent(opt['schedule_off_hours'])} |",
        f"| Las tres combinadas frente al Compose actual | {percent(opt['combined'])} |",
        (
            f"| Lote horario efímero frente a Fargate siempre encendido | "
            f"{percent(opt['batch_vs_always_on'])} |"
        ),
        "",
        "Diferencia de precio de México (Central) frente a US East (N. Virginia): "
        + ", ".join(
            f"{instance} {value * 100:+.1f} %"
            for instance, value in opt["region_mx_vs_us_east_1"].items()
        )
        + ".",
        "",
    ]
    return "\n".join(lines)


def plot_capacity(summary: dict, path: Path) -> None:
    fig, (ax_rps, ax_p95) = plt.subplots(1, 2, figsize=(13, 5))
    for run in summary["capacity_runs"]:
        steps = pd.DataFrame(run["steps"])
        ax_rps.plot(steps["users"], steps["rps"], marker="o", label=run["label"])
        ax_p95.plot(steps["users"], steps["p95_ms"], marker="o", label=run["label"])
    slo = summary["parameters"]["slo_p95_ms"]
    ax_p95.axhline(slo, color="gray", linestyle="--", label=f"SLO p95 = {slo:.0f} ms")
    for ax, title, ylabel in (
        (ax_rps, "Peticiones por segundo", "req/s"),
        (ax_p95, "Latencia p95", "ms"),
    ):
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 32], labels=["1", "2", "4", "8", "16", "32"])
        ax.set_xlabel("Usuarios concurrentes (sin tiempo de espera)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
    ax_p95.set_yscale("log")
    ax_p95.legend(fontsize=8)
    fig.suptitle("Capacidad de la API MiBici por configuración", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_costs(summary: dict, path: Path) -> None:
    options = summary["options"]
    app_costs = {
        cost["option"]: cost for cost in summary["scenario_costs"] if cost["scenario"] == "app"
    }
    labels = [option["label"].replace(": ", ":\n").replace(" en ", "\nen ") for option in options]
    fig, (ax_month, ax_unit) = plt.subplots(1, 2, figsize=(14, 6))
    monthly = [app_costs[option["key"]]["monthly_usd"] for option in options]
    per_million = [option["usd_per_million_planned"] for option in options]
    for ax, values, title, fmt in (
        (ax_month, monthly, "Costo mensual con la demanda de la app (USD)", "${:.2f}"),
        (ax_unit, per_million, "USD por millón de inferencias al 70 % de capacidad", "${:.3f}"),
    ):
        bars = ax.barh(labels, values, color="tab:blue")
        for bar, value in zip(bars, values):
            ax.text(
                value,
                bar.get_y() + bar.get_height() / 2,
                " " + fmt.format(value),
                va="center",
                fontsize=9,
            )
        ax.set_title(title)
        ax.invert_yaxis()
        ax.set_xlim(0, max(values) * 1.25)
        ax.grid(axis="x", alpha=0.3)
    ax_unit.set_yticklabels([])
    fig.suptitle(
        f"Opciones de despliegue en AWS {summary['pricing']['region_name']}", fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "finops_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "finops_report.md").write_text(render_markdown(summary), encoding="utf-8")
    plot_capacity(summary, output_dir / "capacity_by_configuration.png")
    plot_costs(summary, output_dir / "cost_by_deployment_option.png")


@app.command()
def main(
    capacity_dir: Path = CAPACITY_DIR,
    pricing_path: Path = PRICING_PATH,
    hourly_path: Path = HOURLY_DATA_PATH,
    output_dir: Path = OUTPUT_DIR,
    slo_p95_ms: float = SLO_P95_MS,
):
    """Calcula el costo por inferencia y escribe el reporte en reports/finops/."""
    if not hourly_path.exists():
        raise typer.BadParameter(
            f"No existe {hourly_path}. Generarlo con el pipeline de datos (sección 8 del README)."
        )
    pricing = json.loads(pricing_path.read_text(encoding="utf-8"))
    profile = demand_profile(
        pd.read_parquet(hourly_path, columns=["station_id", "hour", "departures"])
    )
    summary = analyze(capacity_dir, pricing, profile, slo_p95_ms)
    write_outputs(summary, output_dir)
    for option in summary["options"]:
        logger.info(
            f"{option['label']}: {money(option['usd_per_million_planned'])} por millón "
            f"({option['capacity_rps']:.1f} req/s, {money(option['monthly_usd'])}/mes)"
        )
    logger.success(f"Reporte FinOps en {output_dir}")


if __name__ == "__main__":
    app()
