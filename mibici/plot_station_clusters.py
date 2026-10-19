from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "data/raw/nomenclatura_2026_09.csv"
CLUSTERS = ROOT / "reports/clustering/station_clusters.csv"
OUTPUT = ROOT / "reports/figures/clustering_station_map.png"


def main():
    stations = pd.read_csv(CATALOG)
    clusters = pd.read_csv(CLUSTERS)

    stations["id"] = pd.to_numeric(stations["id"], errors="coerce")
    stations["latitude"] = pd.to_numeric(
        stations["latitude"], errors="coerce"
    )
    stations["longitude"] = pd.to_numeric(
        stations["longitude"], errors="coerce"
    )
    clusters["station_id"] = pd.to_numeric(
        clusters["station_id"], errors="coerce"
    )

    data = stations.merge(
        clusters[["station_id", "cluster"]],
        left_on="id",
        right_on="station_id",
        how="inner",
    ).dropna(subset=["latitude", "longitude", "cluster"])

    print(f"Estaciones con coordenadas y clúster: {len(data)}")
    print(f"Estaciones sin coincidencia: {len(stations) - len(data)}")

    if data.empty:
        raise ValueError("No hay estaciones coincidentes para graficar.")

    fig, ax = plt.subplots(figsize=(10, 10))

    for cluster in sorted(data["cluster"].unique()):
        subset = data[data["cluster"] == cluster]
        ax.scatter(
            subset["longitude"],
            subset["latitude"],
            s=18,
            alpha=0.75,
            label=f"Clúster {cluster}",
        )

    ax.set_title("Clústeres de estaciones MiBici")
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.legend()
    ax.grid(alpha=0.2)
    ax.set_aspect("equal", adjustable="datalim")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Mapa guardado en: {OUTPUT}")


if __name__ == "__main__":
    main()
