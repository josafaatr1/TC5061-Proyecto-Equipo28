# Entregable 8: datos verificados para redactar el informe

Este archivo reúne insumos, no es el informe final. No convertir resultados retrospectivos en afirmaciones de producción ni completar cifras o identidades no verificadas.

## Contexto y objetivo

MiBici P2 clasifica presión proxy de bicicletas o anclajes por estación-hora usando flujos esperados en tres horas y capacidad de estación. Son proxies, no mediciones reales de inventario o disponibilidad. Se monitorean entradas y etiquetas, y se comparan dos versiones del clasificador de bicicletas.

| Conjunto | Intervalo de 2025 | Ventanas P2 | Uso |
|---|---|---:|---|
| Reference / train | Enero-agosto | 2,137,182 | Entrenamiento v1 y referencia Evidently. |
| Validation histórica | Septiembre-octubre | 536,904 | Selección previa de hiperparámetros v1; incorporada al entrenamiento v2. |
| Current / test | Noviembre-diciembre | 536,245 | Drift y comparación retrospectiva de versiones. |
| Train v2 | Enero-octubre | 2,674,086 | Unión de las dos primeras particiones; sin ventanas entre fronteras. |

Las 15 entradas incluyen calendario, nueve rezagos de flujo y dos medias móviles. Se construyen con información disponible antes de la hora de predicción. Las etiquetas usan umbrales de presión congelados con train, 0.3684210526315789 para cada objetivo. El catálogo de septiembre de 2026 se aplica retrospectivamente a 2025.

## Drift y calidad

Evidently 0.7.23 compara 17 columnas: 15 entradas y dos etiquetas. Se detectan 3/17, 17.647 %, por debajo del umbral global de 50 %. Por eso aparece «Dataset Drift is NOT detected», aunque hay drift por variable. Las 11 entradas numéricas usan Wasserstein normalizado; las cuatro entradas de calendario usan Jensen-Shannon, ambos con umbral 0.1. Las etiquetas usan Z de proporciones con umbral nominal 0.05.

| Variable | Método | Puntaje | Drift |
|---|---|---:|---|
| `hour_of_day` | Jensen-Shannon | 0.001885 | No |
| `day_of_week` | Jensen-Shannon | 0.020745 | No |
| `month` | Jensen-Shannon | 0.832555 | Sí |
| `is_weekend` | Jensen-Shannon | 0.006063 | No |
| `departures_lag_1h` | Wasserstein normalizado | 0.033312 | No |
| `departures_lag_24h` | Wasserstein normalizado | 0.029527 | No |
| `departures_lag_168h` | Wasserstein normalizado | 0.006178 | No |
| `arrivals_lag_1h` | Wasserstein normalizado | 0.029953 | No |
| `arrivals_lag_24h` | Wasserstein normalizado | 0.026416 | No |
| `arrivals_lag_168h` | Wasserstein normalizado | 0.003948 | No |
| `net_flow_lag_1h` | Wasserstein normalizado | 0.043012 | No |
| `net_flow_lag_24h` | Wasserstein normalizado | 0.040863 | No |
| `net_flow_lag_168h` | Wasserstein normalizado | 0.023511 | No |
| `departures_rolling_mean_24h` | Wasserstein normalizado | 0.059847 | No |
| `departures_rolling_mean_168h` | Wasserstein normalizado | 0.051211 | No |
| `target_bike_pressure` | Z de proporciones | p < 0.001 nominal | Sí |
| `target_dock_pressure` | Z de proporciones | p < 0.001 nominal | Sí |

El mes tiene categorías sin solapamiento: 1-8 en reference y 11-12 en current. La prevalencia de bicicletas cae de 4.3383 % a 3.0881 %, -1.2501 puntos porcentuales y -28.82 % relativo. Anclajes cae de 4.0989 % a 3.4257 %, -0.6733 puntos y -16.43 % relativo. El valor cero guardado por Evidently para ambos p-values refleja precisión numérica, no probabilidad exactamente cero. Las ventanas se solapan y hay dependencia temporal y por estación, por lo que los p-values nominales no son inferencia independiente.

| Calidad | Reference | Current |
|---|---:|---:|
| Filas | 2,137,182 | 536,245 |
| Valores faltantes | 284,499 | 771 |
| Columnas vacías | 0 | 0 |
| Columnas constantes | 0 | 0 |
| Columnas duplicadas | 0 | 0 |
| Variables casi constantes | 2 | 2 |

Los faltantes aparecen en rezagos y medias por historia insuficiente; las etiquetas son casi constantes por desbalance. Esto no descarta todos los errores de ingesta. Los resultados completos están en [JSON](../reports/evidently/drift_summary.json), [HTML de drift](../reports/evidently/data_drift_report.html), [HTML de calidad](../reports/evidently/data_quality_report.html) e [interpretación](drift_interpretation.md).

## Comparación principal de bicicletas

| Elemento | A | B |
|---|---|---|
| Modelo registrado | `mibici-p2-bike-pressure` v1 | `mibici-p2-bike-pressure` v2 |
| Estado | `champion` | Challenger, sin cambio de alias |
| Run de entrenamiento | `d306fe5f7f34496a94f8cc3d0db2f62b` | `d2df45da747c4241a8ff657417aa8332` |
| Modelo MLflow | `m-c65129c4cf4d4744b31f026ab2a4c1c1` | `m-d632547af1134d33b65648440c2a918b` |

Ambos son `HistGradientBoostingClassifier`, con clases balanceadas, learning rate 0.2, 63 hojas, 100 iteraciones y semilla 42. A se entrenó en enero-agosto y su configuración se seleccionó con septiembre-octubre. B conserva esa configuración y se reentrena con enero-octubre. Noviembre-diciembre no se usó para entrenar, ajustar hiperparámetros ni calibrar B; sin embargo, el diseño de la receta se decidió después de examinar ese periodo. A y B se evaluaron sobre exactamente las mismas filas y etiquetas de current, con umbral de decisión 0.5. Run de comparación: `9e2ad353a2204ead83043b2e81426e9a`.

| Métrica | A v1 | B v2 | B menos A | Cambio relativo |
|---|---:|---:|---:|---:|
| Average Precision | 0.262416 | 0.277382 | +0.014966 | +5.70 % |
| Precision | 0.118485 | 0.117357 | -0.001128 | -0.95 % |
| Recall | 0.817452 | 0.835085 | +0.017633 | +2.16 % |
| F1 | 0.206970 | 0.205793 | -0.001177 | -0.57 % |
| TN | 418,971 | 415,677 | -3,294 | -0.79 % |
| FP | 100,714 | 104,008 | +3,294 | +3.27 % |
| FN | 3,023 | 2,731 | -292 | -9.66 % |
| TP | 13,537 | 13,829 | +292 | +2.16 % |
| Prevalencia | 0.030881 | 0.030881 | 0 | 0 % |

Los valores completos y huellas están en el [CSV](../reports/ab_testing/p2_bike_registered_comparison.csv); el diseño, la comparación exploratoria anterior y comandos están en [ab_testing_summary.md](ab_testing_summary.md).

## Recomendación y limitaciones

Mantener v1 provisionalmente. B mejora AP y recall, pero produce 3,294 falsas alarmas adicionales y reduce ligeramente precision y F1. No se dispone de costos operativos ni intervalos de confianza, y current ya se había examinado. Es una comparación offline pareada, no un A/B aleatorizado ni una evaluación confirmatoria independiente.

Proponer monitoreo mensual y revisar trimestralmente candidatos. Como señales iniciales no validadas: variación de prevalencia de 20 % relativo o caída de AP/precision de 10 % relativo durante dos cortes comparables consecutivos; investigar, no promover automáticamente. Para promoción: validar en periodo temporal posterior no examinado, definir métricas, recall mínimo y presupuesto de alertas, revisar calibración y costos, obtener aprobación humana y conservar v1 para rollback. El monitoreo parametrizable y CT automático todavía no están implementados. Véase [recomendación completa](continuous_training_recommendation.md).

Las etiquetas proxy utilizan capacidades de 2026 para datos de 2025; no representan inventario real. Hay estacionalidad, ventanas solapadas, dependencia por estación, una sola evaluación temporal y baja precision. No inferir causalidad del drift ni superioridad estadística de v2. La evaluación usa `>= 0.5`, mientras la API usa `model.predict()` y clasifica empates exactos de otra forma.

## Imágenes y tablas para Word

- [Resumen Evidently de drift](evidence/evidently-data-drift.png).
- [Resumen Evidently de calidad](evidence/evidently-data-quality.png).
- [Variables con drift y pruebas](../reports/figures/entrega8/drift_variables.png).
- [Prevalencia de etiquetas](../reports/figures/entrega8/prevalencia_etiquetas.png).
- [Métricas comparativas A/B](../reports/figures/entrega8/ab_metricas_bicicletas.png).
- [Matrices de confusión A/B](../reports/figures/entrega8/ab_matrices_confusion_bicicletas.png).
- [CSV A/B definitivo](../reports/ab_testing/p2_bike_registered_comparison.csv); la tabla Markdown está arriba y en [ab_testing_summary.md](ab_testing_summary.md).

Las cuatro figuras nuevas se generan con `python -m mibici.monitoring.entrega8_figures` a partir del JSON y CSV guardados.

## Referencias verificables

- Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M. y Bouchachia, A. (2014). *A survey on concept drift adaptation*. ACM Computing Surveys, 46(4). https://doi.org/10.1145/2523813
- Saito, T. y Rehmsmeier, M. (2015). *The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets*. PLOS ONE, 10(3), e0118432. https://doi.org/10.1371/journal.pone.0118432
- Scikit-learn. *HistGradientBoostingClassifier*. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html
- MiBici. *Datos abiertos*. https://www.mibici.net/es/datos-abiertos/

## Pruebas y colaboración

- Pruebas dirigidas: 55 aprobadas; suite completa: 294 aprobadas, 2 omitidas, 117 advertencias.
- `ruff check .`, `ruff format --check .`, `git diff --check` y `python -m pip check`: aprobados.
- Los reportes de Evidently se regeneraron; las decisiones por variable coincidieron con el resumen existente.
- El historial contiene contribuciones de varios integrantes en #25, #26 y #27. El PR de esta rama y su revisión deben añadirse como evidencia cuando se realicen manualmente.
- Los datos, modelos y base MLflow no están en Git: preservar o exportar los artefactos para que el equipo pueda reproducirlos.
