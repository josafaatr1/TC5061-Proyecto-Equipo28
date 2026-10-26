# Evaluación final en test — MiBici 2025 (Equipo 28)

Evaluación de los modelos ya seleccionados y registrados (`@champion`) en **noviembre-diciembre de 2025**. Los modelos se entrenaron con enero-agosto y se seleccionaron con septiembre-octubre; no se reentrenaron, no se ajustó nada con test y los `champion` no cambiaron (siguen en la versión 1).

## 1. Resultados

Umbral de decisión de P2: 0.5, el original. «Val» son las métricas recalculadas con este mismo código y coinciden con las del registro (tolerancia relativa de 1e-6).

### P1 — salidas por estación-hora (`mibici-p1-departures@champion`, v1)

| Modelo | Val MAE | Val RMSE | Test MAE | Test RMSE |
|---|---|---|---|---|
| `hist_gradient_boosting_poisson` (lr 0.2, 63 hojas) | 0.8596 | 1.5649 | **0.8121** | **1.4351** |
| Estacional 168 h (baseline) | 1.0383 | 2.0524 | 1.0018 | 1.9206 |
| Media por hora (baseline) | 1.6672 | 2.7013 | 1.5897 | 2.4857 |

Filas evaluadas: 537,472 en validation y 536,813 en test (las que tienen rezago semanal).

### P2 — presión de bicicletas (`mibici-p2-bike-pressure@champion`, v1)

| Modelo | Conjunto | AP | Precision | Recall | F1 | Prevalencia |
|---|---|---|---|---|---|---|
| `hist_gradient_boosting_balanced` | Val | 0.3242 | 0.1478 | 0.8410 | 0.2515 | 0.0408 |
| `hist_gradient_boosting_balanced` | **Test** | **0.2624** | 0.1185 | 0.8175 | 0.2070 | 0.0309 |
| Dummy prior (baseline) | Val | 0.0408 | 0 | 0 | 0 | 0.0408 |
| Dummy prior (baseline) | **Test** | 0.0309 | 0 | 0 | 0 | 0.0309 |

Matriz de confusión del champion en test (`[[TN, FP], [FN, TP]]`): `[[418,971, 100,714], [3,023, 13,537]]`.

### P2 — presión de anclajes (`mibici-p2-dock-pressure@champion`, v1)

| Modelo | Conjunto | AP | Precision | Recall | F1 | Prevalencia |
|---|---|---|---|---|---|---|
| `hist_gradient_boosting_balanced` | Val | 0.3882 | 0.1538 | 0.8490 | 0.2604 | 0.0421 |
| `hist_gradient_boosting_balanced` | **Test** | **0.3646** | 0.1355 | 0.8412 | 0.2334 | 0.0343 |
| Dummy prior (baseline) | Val | 0.0421 | 0 | 0 | 0 | 0.0421 |
| Dummy prior (baseline) | **Test** | 0.0343 | 0 | 0 | 0 | 0.0343 |

Matriz de confusión del champion en test: `[[419,279, 98,596], [2,917, 15,453]]`.

Ventanas evaluadas en P2: 536,904 en validation y 536,245 en test, en ambos objetivos.

## 2. Comparación con los baselines

Los baselines se evaluaron **sin reentrenar**: se cargaron los modelos que quedaron registrados en el mismo run padre (`P1-search` y `P2-search`) que el champion.

| Problema | Mejora del champion sobre el mejor baseline en validation | Mejora en test |
|---|---|---|
| P1 (MAE frente al estacional) | 17.2 % | 18.9 % |
| P2 bicicletas (AP frente a la prevalencia) | 7.9 veces | 8.5 veces |
| P2 anclajes (AP frente a la prevalencia) | 9.2 veces | 10.6 veces |

## 3. Interpretación

- **P1 generaliza.** El error absoluto baja en test (MAE 0.8596 → 0.8121), pero también baja en los dos baselines (el estacional pasa de 1.0383 a 1.0018). La caída es general y no indica un modelo mejor: puede deberse a un periodo con otra dinámica de demanda. Lo relevante es que la ventaja sobre el mejor baseline no se reduce (17.2 % → 18.9 %).
- **P2 conserva su capacidad de ordenar, con menos aciertos absolutos.** El AP baja (bicicletas −0.062; anclajes −0.024) y la precisión también, pero la prevalencia de eventos cae (4.1 % → 3.1 % y 4.2 % → 3.4 %) y ambas métricas dependen de ella. El recall casi no cambia (0.841 → 0.818 y 0.849 → 0.841) y la ventaja sobre el baseline crece. Con umbral 0.5 el modelo sigue alertando cerca del 21.3 % de las estaciones-hora, con una precisión de 0.12 a 0.14: sirve para priorizar revisión, no como alarma automática.
- **Anclajes generaliza mejor que bicicletas** (AP −6 % frente a −19 % en términos relativos).

## 4. Limitaciones

- Una sola partición temporal y una sola semilla: no hay intervalos de confianza. Las diferencias pequeñas entre validation y test no son concluyentes.
- Test son dos meses y diciembre no aparece en entrenamiento (enero-agosto); parte de la diferencia puede ser estacionalidad.
- Las etiquetas de P2 son proxies de flujo con una capacidad del catálogo de 2026-09, no inventario real. Los umbrales de presión (0.3684 en ambos objetivos) son los calculados con train; con test, la prevalencia cambia y por eso baja la precisión.
- El umbral de decisión 0.5 no se calibró; su elección operativa sigue pendiente.
- Los modelos no se reajustaron con validation: se evalúan tal como se seleccionaron.
- P3 (clustering) no se evalúa en test.

## 5. Controles contra fugas de datos

- Los umbrales de P2 se leen de los parámetros del run de origen (calculados solo con train) y nunca se recalculan con test.
- Las ventanas y etiquetas de test salen solo de noviembre-diciembre; los predictores usan solo historial anterior a cada hora (rezagos y medias móviles).
- Antes de evaluar se verificó que coinciden: la huella del dataset de enero-octubre con la de los modelos (`836a24d9…`), el SHA-256 del catálogo (`206e45fe…`) y el de las etiquetas de train/validation (`47009f11…`).
- Test no se usó para elegir hiperparámetros ni modelos.

## 6. Reproducción

Con el servidor MLflow activo y los champion registrados:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
export MLFLOW_EXPERIMENT_NAME=mibici-equipo28
# Solo Windows: export PYTHONUTF8=1  (PowerShell: $env:PYTHONUTF8=1)
mlflow run . -e evaluate --env-manager local
```

Duración observada: 84 s. Pruebas: `python -m pytest tests/test_evaluate.py tests/test_p2_targets.py`.

## 7. Identificadores y evidencias en MLflow (experimento `mibici-equipo28`)

| Elemento | Run |
|---|---|
| Padre `test-evaluation` (artefacto `reports/test_evaluation.csv`) | `003f4b2409414a05b41cde4a933a64ad` |
| `P1-test-champion` | `589954b4116744da83132f80eb84d385` |
| `P1-test-seasonal-weekly` | `2dad2246ee6545889a102088445b856e` |
| `P1-test-dummy-mean` | `1cd241b8a4b147f4b4df4167ee3fb281` |
| `P2-bike-pressure-test-champion` | `99430803c2e440808683eaca1928f313` |
| `P2-bike-pressure-test-dummy-prior` | `0985afe18d1b487e9f518e2b990eab94` |
| `P2-dock-pressure-test-champion` | `842eefafd30649d380fcf043d591da76` |
| `P2-dock-pressure-test-dummy-prior` | `476babda8b46483a98e38b00b212715a` |

Procedencia de los champion (run donde se seleccionaron): P1 `ae2ce24e8f6b43b3b4e069caa1f22733`, bicicletas `d306fe5f7f34496a94f8cc3d0db2f62b` y anclajes `81f93c6f160e4f44afe3db913eac5030`, todos hijos de `P1-search` o `P2-search`.

Cada run hijo registra: métricas `val_*` y `test_*`, el parámetro `test_period` (2025-11-01/2026-01-01), y en P2 el umbral de decisión y los umbrales de presión; y los tags `stage_of_work=test`, `evaluated_role`, `evaluated_model_uri`, `evaluated_model_name` y `evaluated_model_version` (champion), `source_run_id`, `source_run_name`, `train_validation_data_version`, `full_year_data_version` (`ad5b310cf014800e6ba0ab16032b74b0ab0909a2a942aea46a20e1646839ec68`), el SHA-256 del catálogo y de las etiquetas, y el commit de Git.
