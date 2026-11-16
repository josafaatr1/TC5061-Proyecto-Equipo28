"""Crear figuras de la entrega 8 a partir de resultados guardados."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    summary = json.loads(
        Path("reports", "evidently", "drift_summary.json").read_text(encoding="utf-8")
    )
    comparison = pd.read_csv(Path("reports", "ab_testing", "p2_bike_registered_comparison.csv"))
    comparison = comparison.set_index("role").loc[["A", "B"]]
    output = Path("reports", "figures", "entrega8")
    output.mkdir(parents=True, exist_ok=True)
    colors = ("#245678", "#aa4b22")
    columns = {row["column"]: row for row in summary["columns"]}

    fig, (ax, text_ax) = plt.subplots(
        1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1, 1.25]}
    )
    month = columns["month"]
    ax.barh(["Mes"], [month["score"]], color=colors[0])
    ax.axvline(month["threshold"], color=colors[1], linestyle="--", label="Umbral: 0.1")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Distancia Jensen-Shannon")
    ax.text(month["score"] + 0.015, 0, f"{month['score']:.4f}", va="center")
    ax.legend(loc="lower right")
    text_ax.axis("off")
    lines = ["Etiquetas proxy: prueba Z de proporciones"]
    for column, label in (
        ("target_bike_pressure", "Bicicletas"),
        ("target_dock_pressure", "Anclajes"),
    ):
        score = columns[column]["score"]
        displayed = "p < 0.001" if score < 0.001 else f"p = {score:.3f}"
        lines.append(f"{label}: {displayed} (nominal)")
    lines.append("Ventanas solapadas: p nominal no ajustado por dependencia.")
    text_ax.text(0.0, 0.65, "\n\n".join(lines), va="center", fontsize=10)
    fig.suptitle("Variables con drift: reference frente a current")
    fig.tight_layout()
    fig.savefig(output / "drift_variables.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(2)
    prevalence = summary["prevalence"]
    target_columns = ("target_bike_pressure", "target_dock_pressure")
    for offset, period, color in ((-0.18, "reference", colors[0]), (0.18, "current", colors[1])):
        values = [100 * prevalence[column][period] for column in target_columns]
        bars = ax.bar(x + offset, values, width=0.36, label=period.capitalize(), color=color)
        ax.bar_label(bars, fmt="%.2f %%", padding=3)
    ax.set_xticks(x, ["Bicicletas", "Anclajes"])
    ax.set_ylabel("Ventanas positivas (%)")
    ax.set_ylim(0, 5.4)
    ax.set_title("Prevalencia de etiquetas proxy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "prevalencia_etiquetas.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    metrics = ["average_precision", "precision", "recall", "f1"]
    x = np.arange(len(metrics))
    for offset, role, color in ((-0.18, "A", colors[0]), (0.18, "B", colors[1])):
        values = comparison.loc[role, metrics].astype(float).to_numpy()
        bars = ax.bar(
            x + offset,
            values,
            width=0.36,
            label=f"{role} (v{comparison.loc[role, 'version']})",
            color=color,
        )
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_xticks(x, ["AP", "Precision", "Recall", "F1"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Valor")
    ax.set_title("P2 bicicletas: comparación retrospectiva en current")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "ab_metricas_bicicletas.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))
    for ax, role in zip(axes, ("A", "B"), strict=True):
        row = comparison.loc[role]
        counts = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]], dtype=int)
        ratios = counts / counts.sum(axis=1, keepdims=True)
        ax.imshow(ratios, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    f"{counts[i, j]:,}\n{ratios[i, j]:.1%}",
                    ha="center",
                    va="center",
                    color="white" if ratios[i, j] > 0.55 else "black",
                )
        ax.set_xticks([0, 1], ["Negativa", "Positiva"])
        ax.set_yticks([0, 1], ["Negativa", "Positiva"])
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Etiqueta real")
        ax.set_title(f"{role} (v{row['version']})")
    fig.suptitle("Matriz de confusión: conteos y porcentaje por clase real")
    fig.tight_layout()
    fig.savefig(output / "ab_matrices_confusion_bicicletas.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
