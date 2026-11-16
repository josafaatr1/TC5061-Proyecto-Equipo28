# Insumo técnico: drift y calidad de P2

## Conjuntos y alcance

- **Reference:** train, enero-agosto de 2025, 2,137,182 ventanas.
- **Current:** test retrospectivo, noviembre-diciembre de 2025, 536,245 ventanas. No es telemetría de producción.
- Se comparan 15 predictores y dos etiquetas proxy con umbrales de presión de train: 0.368421 para ambos objetivos. El catálogo usado es el de septiembre de 2026, con capacidades retrospectivas.

## Hallazgos

| Variable | Método | Resultado | Magnitud |
|---|---|---|---|
| `month` | Distancia Jensen-Shannon, umbral 0.1 | 0.8326, drift | Reference: meses 1-8; current: meses 11-12, sin solapamiento. |
| `target_bike_pressure` | Z de proporciones, p < 0.05 | p < 0.001 nominal | 4.3383 % a 3.0881 %, diferencia -1.2501 puntos porcentuales, -28.82 % relativo. |
| `target_dock_pressure` | Z de proporciones, p < 0.05 | p < 0.001 nominal | 4.0989 % a 3.4257 %, diferencia -0.6733 puntos porcentuales, -16.43 % relativo. |

El resumen estructurado completo de las 17 variables, sus pruebas, umbrales y puntuaciones está en [drift_summary.json](../reports/evidently/drift_summary.json). Las 11 variables numéricas de flujo usan Wasserstein normalizado; ninguna supera 0.1. Las etiquetas tienen p-value redondeado a cero por precisión numérica, no una probabilidad literalmente nula.

El titular global de Evidently indica que no detecta drift del dataset porque 3/17 = 17.6 % está por debajo del umbral global de 50 %. No implica ausencia de cambios relevantes. La prueba Z nominal no contempla dependencia temporal o ventanas solapadas.

En calidad, el HTML registra 284,499 valores faltantes en reference y 771 en current, asociados a historia insuficiente en rezagos y medias. No hay columnas vacías, constantes o duplicadas; las etiquetas son casi constantes por desbalance. Estos datos no prueban por sí solos que la ingesta carezca de errores.

## Desempeño y cautelas

El champion de bicicletas pasa de AP 0.3242 en validation a 0.2624 en current; el de anclajes, de 0.3882 a 0.3646. La menor prevalencia puede contribuir, pero estos datos no identifican una causa ni prueban que el ranking sea igual de bueno. La precisión a umbral 0.5 en current es 0.1185 y 0.1355, respectivamente. Véanse [evaluación histórica](../reports/test_evaluation.md) y [comparación de variantes](ab_testing_summary.md).

Fuentes y figuras: [reporte de drift](../reports/evidently/data_drift_report.html), [reporte de calidad](../reports/evidently/data_quality_report.html), [imagen de drift](evidence/evidently-data-drift.png) y [imagen de calidad](evidence/evidently-data-quality.png). La interpretación detallada está en [drift_interpretation.md](drift_interpretation.md).

La comparación principal posterior entre v1 y v2 de bicicletas está en [ab_testing_summary.md](ab_testing_summary.md). Figuras reproducibles adicionales: [variables con drift](../reports/figures/entrega8/drift_variables.png) y [prevalencias](../reports/figures/entrega8/prevalencia_etiquetas.png).
