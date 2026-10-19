
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from mibici.config import (
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    RANDOM_SEED,
)


def main():
    input_path = PROCESSED_DATA_DIR / "station_hourly_2025_complete.parquet"
    output_dir = REPORTS_DIR / "clustering"
    figures_dir = FIGURES_DIR

    if not input_path.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset: {input_path}. "
            "Ejecuta primero el pipeline de datos."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Cargando datos...")
    df = pd.read_parquet(input_path)

    required = {
        "station_id",
        "departures",
        "arrivals",
        "net_flow",
        "total_activity",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")

    logger.info("Calculando indicadores por estación...")
    features = df.groupby("station_id").agg(
        total_departures=("departures", "sum"),
        total_arrivals=("arrivals", "sum"),
        mean_departures=("departures", "mean"),
        mean_arrivals=("arrivals", "mean"),
        mean_activity=("total_activity", "mean"),
        total_activity=("total_activity", "sum"),
        mean_net_flow=("net_flow", "mean"),
        std_activity=("total_activity", "std"),
    ).fillna(0)

    if len(features) < 3:
        raise ValueError("Se necesitan al menos 3 estaciones para el clustering.")

    features.to_csv(output_dir / "station_features.csv")

    # Reducir la asimetría de los indicadores antes de estandarizarlos.
    X = features.clip(lower=0).copy()
    signed_columns = ["mean_net_flow"]
    for column in X.columns:
        if column not in signed_columns:
            X[column] = X[column].clip(lower=0).map(lambda value: __import__("numpy").log1p(value))

    X["mean_net_flow"] = features["mean_net_flow"].map(
        lambda value: __import__("numpy").sign(value)
        * __import__("numpy").log1p(abs(value))
    )

    # Eliminar indicadores constantes para evitar problemas de escalado.
    X = X.loc[:, X.nunique() > 1]
    if X.shape[1] == 0:
        raise ValueError("No hay indicadores con variación suficiente para agrupar.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    logger.info("Evaluando diferentes cantidades de grupos...")
    results = []
    max_k = min(8, len(features) - 1)

    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = model.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        results.append({"k": k, "silhouette_score": score})
        logger.info(f"k={k}: silhouette={score:.4f}")

    if not results:
        raise ValueError("No fue posible evaluar cantidades de grupos.")

    evaluation = pd.DataFrame(results)
    evaluation.to_csv(output_dir / "evaluation_metrics.csv", index=False)

    best_k = int(
        evaluation.loc[evaluation["silhouette_score"].idxmax(), "k"]
    )
    logger.info(f"Cantidad de grupos seleccionada: {best_k}")

    model = KMeans(n_clusters=best_k, random_state=RANDOM_SEED, n_init=10)
    labels = model.fit_predict(X_scaled)

    station_clusters = features.copy()
    station_clusters["cluster"] = labels
    station_clusters.to_csv(output_dir / "station_clusters.csv")

    profiles = station_clusters.groupby("cluster").mean(numeric_only=True)
    profiles.to_csv(output_dir / "cluster_profiles.csv")

    # Gráfica de evaluación.
    plt.figure(figsize=(8, 5))
    plt.plot(
        evaluation["k"],
        evaluation["silhouette_score"],
        marker="o",
    )
    plt.xlabel("Número de grupos (k)")
    plt.ylabel("Silhouette score")
    plt.title("Evaluación de clustering")
    plt.tight_layout()
    plt.savefig(figures_dir / "clustering_silhouette.png", dpi=150)
    plt.close()

    # Proyección de estaciones en dos dimensiones.
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    coordinates = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=labels,
        cmap="tab10",
        s=25,
        alpha=0.8,
    )
    plt.xlabel("Componente principal 1")
    plt.ylabel("Componente principal 2")
    plt.title(f"Estaciones agrupadas con K-Means (k={best_k})")
    plt.legend(*scatter.legend_elements(), title="Grupo")
    plt.tight_layout()
    plt.savefig(figures_dir / "clustering_pca.png", dpi=150)
    plt.close()

    logger.success("Clustering terminado.")
    logger.info(f"Resultados guardados en: {output_dir}")
    logger.info(f"Gráficas guardadas en: {figures_dir}")


if __name__ == "__main__":
    main()
