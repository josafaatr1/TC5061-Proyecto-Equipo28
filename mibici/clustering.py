"""Agrupamiento de estaciones por su perfil de uso (P3).

Usa solo enero-octubre de 2025, igual que P1 y P2: noviembre y diciembre
quedan reservados como test.

    python -m mibici.clustering
"""

from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from mibici.config import (
    FIGURES_DIR,
    P3_MAX_K,
    P3_N_INIT,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
)
from mibici.modeling.p2_targets import VALIDATION_END, YEAR_START

INPUT_PATH = PROCESSED_DATA_DIR / "station_hourly_2025_complete.parquet"
OUTPUT_DIR = REPORTS_DIR / "clustering"
REQUIRED_COLUMNS = {"station_id", "hour", "departures", "arrivals", "net_flow", "total_activity"}
SIGNED_COLUMNS = ("mean_net_flow",)
MAX_K = P3_MAX_K
N_INIT = P3_N_INIT


@dataclass(frozen=True)
class ClusteringResult:
    """Indicadores, evaluación de k y modelo final del agrupamiento."""

    features: pd.DataFrame
    evaluation: pd.DataFrame
    best_k: int
    model: Pipeline
    labels: np.ndarray


def read_hourly(input_path: Path = INPUT_PATH) -> pd.DataFrame:
    """Leer el dataset horario de enero a octubre de 2025."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset: {input_path}. Ejecuta primero el pipeline de datos."
        )

    return pd.read_parquet(
        input_path,
        filters=[("hour", ">=", YEAR_START), ("hour", "<", VALIDATION_END)],
    )


def build_station_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcular los indicadores de uso de cada estación."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")

    # Variables generales por estación.
    features = (
        df.groupby("station_id")
        .agg(
            total_departures=("departures", "sum"),
            total_arrivals=("arrivals", "sum"),
            mean_departures=("departures", "mean"),
            mean_arrivals=("arrivals", "mean"),
            mean_activity=("total_activity", "mean"),
            total_activity=("total_activity", "sum"),
            mean_net_flow=("net_flow", "mean"),
            std_activity=("total_activity", "std"),
        )
        .fillna(0)
    )

    # Patrones de uso por hora del día.
    hours = pd.to_datetime(df["hour"])
    hourly = (
        df.groupby(["station_id", hours.dt.hour.rename("hour_of_day")])[["departures", "arrivals"]]
        .mean()
        .unstack(fill_value=0)
    )
    hourly.columns = [f"{kind}_hora_{hour:02d}" for kind, hour in hourly.columns]

    # Actividad promedio en días laborales y fines de semana.
    day_type = np.where(hours.dt.dayofweek >= 5, "actividad_fin_semana", "actividad_laboral")
    daily = (
        df.groupby(["station_id", pd.Series(day_type, index=df.index, name="day_type")])[
            "total_activity"
        ]
        .mean()
        .unstack(fill_value=0)
    )

    features = features.join(hourly, how="left").join(daily, how="left").fillna(0)

    if len(features) < 3:
        raise ValueError("Se necesitan al menos 3 estaciones para el clustering.")

    # Eliminar indicadores constantes para evitar problemas de escalado.
    features = features.loc[:, features.nunique() > 1]
    if features.shape[1] == 0:
        raise ValueError("No hay indicadores con variación suficiente para agrupar.")

    return features


def log_transform(X: pd.DataFrame) -> pd.DataFrame:
    """Reducir la asimetría de los indicadores antes de estandarizarlos."""
    X = X.copy()
    for column in X.columns:
        if column in SIGNED_COLUMNS:
            X[column] = np.sign(X[column]) * np.log1p(np.abs(X[column]))
        else:
            X[column] = np.log1p(X[column].clip(lower=0))

    return X


def make_model(k: int) -> Pipeline:
    """Transformación logarítmica, estandarización y K-Means con ``k`` grupos."""
    return Pipeline(
        [
            ("log", FunctionTransformer(log_transform)),
            ("scaler", StandardScaler()),
            ("kmeans", KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=N_INIT)),
        ]
    )


def scaled_features(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Indicadores transformados y estandarizados con un modelo ajustado."""
    return model[:-1].transform(features)


def evaluate_k(features: pd.DataFrame, max_k: int = MAX_K) -> pd.DataFrame:
    """Calcular silhouette y Davies-Bouldin para cada cantidad de grupos."""
    results = []
    for k in range(2, min(max_k, len(features) - 1) + 1):
        model = make_model(k)
        labels = model.fit_predict(features)
        X_scaled = scaled_features(model, features)
        score = silhouette_score(X_scaled, labels)
        db_score = davies_bouldin_score(X_scaled, labels)
        results.append({"k": k, "silhouette_score": score, "davies_bouldin_score": db_score})
        logger.info(f"k={k}: silhouette={score:.4f}, Davies-Bouldin={db_score:.4f}")

    if not results:
        raise ValueError("No fue posible evaluar cantidades de grupos.")

    return pd.DataFrame(results)


def fit_clusters(
    features: pd.DataFrame,
    max_k: int = MAX_K,
    k: int | None = None,
) -> ClusteringResult:
    """Elegir k por silhouette (o usar ``k`` fijo) y ajustar el modelo final."""
    evaluation = evaluate_k(features, max_k)
    if k is None:
        best_k = int(evaluation.loc[evaluation["silhouette_score"].idxmax(), "k"])
    elif k in set(evaluation["k"]):
        best_k = k
    else:
        raise ValueError(f"k={k} debe estar entre 2 y {evaluation['k'].max()}.")
    logger.info(f"Cantidad de grupos seleccionada: {best_k}")

    model = make_model(best_k)
    labels = model.fit_predict(features)

    return ClusteringResult(features, evaluation, best_k, model, labels)


def cluster_profiles(features: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Promedio de los indicadores por grupo."""
    return features.assign(cluster=labels).groupby("cluster").mean(numeric_only=True)


def silhouette_figure(evaluation: pd.DataFrame) -> Figure:
    """Graficar el silhouette score por cantidad de grupos."""
    figure = Figure(figsize=(8, 5), layout="constrained")
    axes = figure.subplots()
    axes.plot(evaluation["k"], evaluation["silhouette_score"], marker="o")
    axes.set_xlabel("Número de grupos (k)")
    axes.set_ylabel("Silhouette score")
    axes.set_title("Evaluación de clustering")

    return figure


def pca_figure(X_scaled: np.ndarray, labels: np.ndarray, k: int) -> Figure:
    """Proyectar las estaciones en dos dimensiones coloreadas por grupo."""
    coordinates = PCA(n_components=2, random_state=RANDOM_SEED).fit_transform(X_scaled)

    figure = Figure(figsize=(8, 6), layout="constrained")
    axes = figure.subplots()
    scatter = axes.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=labels,
        cmap="tab10",
        vmin=0,
        vmax=9,
        s=25,
        alpha=0.8,
    )
    axes.set_xlabel("Componente principal 1")
    axes.set_ylabel("Componente principal 2")
    axes.set_title(f"Estaciones agrupadas con K-Means (k={k})")
    axes.legend(*scatter.legend_elements(), title="Grupo")

    return figure


def main():
    output_dir = OUTPUT_DIR
    figures_dir = FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Cargando datos...")
    df = read_hourly()

    logger.info("Calculando indicadores por estación...")
    features = build_station_features(df)
    features.to_csv(output_dir / "station_features.csv")

    logger.info("Evaluando diferentes cantidades de grupos...")
    result = fit_clusters(features)
    result.evaluation.to_csv(output_dir / "evaluation_metrics.csv", index=False)

    station_clusters = features.assign(cluster=result.labels)
    station_clusters.to_csv(output_dir / "station_clusters.csv")
    cluster_profiles(features, result.labels).to_csv(output_dir / "cluster_profiles.csv")

    silhouette_figure(result.evaluation).savefig(
        figures_dir / "clustering_silhouette.png", dpi=150
    )
    pca_figure(scaled_features(result.model, features), result.labels, result.best_k).savefig(
        figures_dir / "clustering_pca.png", dpi=150
    )

    logger.success("Clustering terminado.")
    logger.info(f"Resultados guardados en: {output_dir}")
    logger.info(f"Gráficas guardadas en: {figures_dir}")


if __name__ == "__main__":
    main()
