# Auditoría técnica del entregable 8

Los estados reflejan la evidencia disponible, no garantizan la calificación. La comparación de noviembre-diciembre es retrospectiva y ese periodo ya fue examinado.

| Criterio | Estado | Evidencia | Observaciones |
|---|---|---|---|
| I1: reference y current | Completo | [Resumen de drift](drift_summary_for_report.md) | Train enero-agosto y test noviembre-diciembre de 2025; current simulado, no producción. |
| I1: reporte drift | Completo | [HTML](../reports/evidently/data_drift_report.html), [JSON](../reports/evidently/drift_summary.json) | Regenerados con catálogo histórico; 3 de 17 variables. |
| I1: calidad y gráficas | Completo | [HTML](../reports/evidently/data_quality_report.html), [captura drift](evidence/evidently-data-drift.png), [captura calidad](evidence/evidently-data-quality.png) | Faltantes de historia y desbalance requieren explicación. |
| I2: variables e interpretación | Completo | [Interpretación](drift_interpretation.md) | El umbral global no descarta drift por variable; Z nominal no considera dependencia. |
| I2: desempeño | Completo | [Evaluación](../reports/test_evaluation.md), [resumen](drift_summary_for_report.md) | Asociación descriptiva, no causalidad demostrada. |
| I3: segunda instancia del mismo modelo | Completo | [Comparación](ab_testing_summary.md), [CSV](../reports/ab_testing/p2_bike_registered_comparison.csv) | B se reentrenó hasta octubre y se registró como v2 de bicicletas, sin mover el alias. |
| I3: comparación en current | Completo para la rúbrica offline | [CSV](../reports/ab_testing/p2_bike_registered_comparison.csv) | Pareada y reproducible, pero retrospectiva, no aleatorizada y sin test independiente para promoción. |
| I3: conclusión de vigencia | Completo como decisión provisional | [Comparación](ab_testing_summary.md) | Mantener A; faltan costos, incertidumbre y validación prospectiva para decidir promoción. |
| I4: recomendación y CT | Completo como propuesta | [Recomendación](continuous_training_recommendation.md) | CT y monitoreo mensual aún no están implementados. |
| I4: claridad y colaboración Git | Parcial | Historial: #25, #26 y #27; [resumen de drift](drift_summary_for_report.md) | Hay commits y merges de distintos colaboradores; documentar revisión/aprobaciones del trabajo actual después del PR manual. |

## Validación efectuada

- `pytest` dirigido: 55 pruebas aprobadas, cuatro advertencias de dependencias de Evidently.
- `pytest -q`: 294 aprobadas, 2 omitidas y 117 advertencias.
- Ruff `check .` y `format --check .`: aprobados.
- Reportes de Evidently regenerados con el catálogo de septiembre. El JSON coincide salvo una diferencia de redondeo de 3.6e-17 en `is_weekend`.
- Comparación offline principal: v1 y v2 evaluadas sobre 536,245 ventanas de bicicletas. Las salidas de v1 coinciden con la evaluación histórica.

## Para mayor solidez

1. Comparar v1 y v2 en un periodo posterior no examinado, con umbral, métrica y costos definidos previamente; incluir incertidumbre mediante bloques temporales o por estación.
2. Implementar y probar periodos parametrizables antes de afirmar que existe monitoreo mensual o CT operativo.
3. Conservar o exportar los modelos y datos fuera de Git de forma accesible al equipo; adjuntar evidencia del PR y revisión después de completarlos manualmente.
