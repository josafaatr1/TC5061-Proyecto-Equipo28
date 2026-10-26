# Entrega 2 — MLflow Project y tracking avanzado (Equipo 28)

Reporte de la búsqueda de hiperparámetros de P1 y P2, la selección del mejor modelo y el Model Registry. Todas las cifras salen de runs reales del experimento `mibici-equipo28` ejecutados con `mlflow run` contra el servidor `http://127.0.0.1:5000` (SQLite). Las cifras son de **validation** (septiembre y octubre de 2025). El **test** (noviembre y diciembre) no se evaluó.

## 1. Qué se agregó

| Pieza | Descripción |
|---|---|
| Entry point `search` | `mlflow run . -e search -P question=p1\|p2`. Reutiliza `run_p1` y `run_p2` (`mibici/modeling/mlflow_runs.py`) con una rejilla de `learning_rate` × `max_leaf_nodes` (`SEARCH_*` en `mibici/config.py`). |
| Jerarquía de runs | Padre `P1-search` o `P2-search`; hijos: los baselines y un candidato por combinación (en P2, uno por combinación y objetivo). |
| Tags de cada hijo | `question`, `model_family`, `stage_of_work` (`baseline`, `training` o `search`), `mlflow.source.git.commit`, `train_validation_data_version`, `model_name` y, en P2, `objective`. |
| Artefactos de cada hijo | `evaluation/` (P1: perfil horario real contra predicho; P2: matriz de confusión y curva Precision-Recall), `manifests/` (`data_manifest.json`, manifiesto de fuentes e inventario con SHA-256), `config/run_config.json` y `model/` con firma y ejemplo de entrada. |
| Resumen del padre | `reports/selection_summary.json` (mejor candidato frente al mejor baseline), tags `selected_run_*` y la tabla `reports/p*_model_comparison.csv`. |

El Model Registry (`mibici/modeling/registry.py`) no cambió: ya elige el mejor hijo del entrenamiento más reciente de cada pregunta por `val_mae` (P1) o `val_average_precision` (P2). Una prueba de regresión confirma que los hijos de una búsqueda compiten entre sí.

## 2. Reproducción desde un clon limpio

Con Python 3.14.4. Los comandos usan la sintaxis de bash; en PowerShell exporta con `$env:NOMBRE="valor"`.

```bash
git clone https://github.com/josafaatr1/TC5061-Proyecto-Equipo28.git
cd TC5061-Proyecto-Equipo28
git switch feature/anthony-mlflow-experiments   # hasta que se integre a main
python3.14 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt

# Terminal aparte, con el mismo ambiente activado:
mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --host 127.0.0.1 --port 5000

# Terminal de trabajo:
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
export MLFLOW_EXPERIMENT_NAME=mibici-equipo28
# Solo Windows: evita errores de codificación al redirigir la salida de mlflow.
# export PYTHONUTF8=1

mlflow run . -e ingest --env-manager local                  # descarga los 12 CSV (393 MB aprox.)
mlflow run . -e p2_catalog --env-manager local              # requiere Internet
mlflow run . -e p2_targets --env-manager local
mlflow run . -e train -P question=p3 --env-manager local
mlflow run . -e search -P question=p1 --env-manager local   # baselines + 9 candidatos
mlflow run . -e search -P question=p2 --env-manager local   # baselines + 9 candidatos por objetivo
mlflow run . -e register --env-manager local                # alias champion
```

`register` va al final porque toma el entrenamiento más reciente de cada pregunta. Si corres `p1_train` o `p2_train` después de `search`, vuelve a correr `search` antes de `register`.

Qué se ejecutó realmente en esta validación (Windows, clon existente con los datos ya descargados): `ingest`, `train -P question=p3`, `search` de P1 y P2 y `register`. **No** se volvió a ejecutar `p2_catalog` ni `p2_targets`: no se descargó nada y se verificó que las etiquetas guardadas se reproducen exactamente (`labels_reproduced_exactly=True`, SHA-256 de las etiquetas `47009f11…`). Tampoco se probó un clon completamente limpio.

Tiempos observados: `ingest` 45 s (reutiliza los CSV), P3 2 min 52 s, búsqueda de P1 7 min 16 s, búsqueda de P2 11 min 40 s y `register` 24 s.

## 3. Datos y particiones

| Elemento | Valor |
|---|---|
| Huella de los CSV originales (`data_version` de `ingest`, `raw_data_version` de los manifiestos) | `3bbd269b92e58c44c787b68ab6e73a1442e59883f2133aae33c486223e15ad1d` (igual a la calculada desde `references/mibici_inventory_2025.csv`) |
| Huella del dataset horario de enero a octubre (`train_validation_data_version`) | `836a24d94f7d493391f7d103390e713fc97e8d2bd4ddb116890a5e216a7cccf0`; idéntica antes y después de correr `ingest` |
| Train / validation P1 | enero–agosto: 2,137,920 filas; septiembre–octubre: 537,640 filas (537,472 evaluadas con rezago semanal) |
| Train / validation P2 | 2,137,182 y 536,904 ventanas |
| Semilla | 42 |

## 4. Resultados de la búsqueda (validation)

La rejilla es `learning_rate` ∈ {0.05, 0.1, 0.2} × `max_leaf_nodes` ∈ {15, 31, 63}; el resto de hiperparámetros no cambia (`max_iter=100`, `min_samples_leaf=50`, `l2_regularization=1.0`). La combinación 0.1 / 31 es la configuración que ya existía y reproduce las cifras del handoff (`val_mae` 0.864087; AP 0.317719 en bicicletas y 0.379491 en anclajes).

### 4.1. P1 — salidas por estación-hora (padre `P1-search`, run `438044ba…`)

Ordenado por `val_mae`. «Mejora» es el porcentaje de reducción de MAE frente al mejor baseline (estacional de 168 h).

| Run | lr | hojas | val_mae | val_rmse | train_mae | Brecha MAE (val − train) | Ajuste (s) | Mejora vs baseline |
|---|---|---|---|---|---|---|---|---|
| `…search-lr0.2-leaf63` | 0.2 | 63 | **0.8596** | 1.5649 | 0.8193 | 0.0403 | 12.3 | 17.2 % |
| `…search-lr0.1-leaf63` | 0.1 | 63 | 0.8601 | 1.5608 | 0.8237 | 0.0364 | 14.1 | 17.2 % |
| `…search-lr0.2-leaf31` | 0.2 | 31 | 0.8609 | **1.5551** | 0.8253 | 0.0356 | 10.9 | 17.1 % |
| `…search-lr0.1-leaf31` | 0.1 | 31 | 0.8641 | 1.5700 | 0.8304 | 0.0337 | 11.5 | 16.8 % |
| `…search-lr0.2-leaf15` | 0.2 | 15 | 0.8648 | 1.5667 | 0.8314 | 0.0334 | 8.6 | 16.7 % |
| `…search-lr0.05-leaf63` | 0.05 | 63 | 0.8674 | 1.5728 | 0.8332 | 0.0343 | 15.5 | 16.5 % |
| `…search-lr0.1-leaf15` | 0.1 | 15 | 0.8708 | 1.5764 | 0.8382 | 0.0326 | 8.6 | 16.1 % |
| `…search-lr0.05-leaf31` | 0.05 | 31 | 0.8736 | 1.5873 | 0.8405 | 0.0331 | 10.9 | 15.9 % |
| `…search-lr0.05-leaf15` | 0.05 | 15 | 0.8828 | 1.6011 | 0.8512 | 0.0316 | 7.4 | 15.0 % |
| `P1-regression-seasonal-168h` (baseline) | — | — | 1.0383 | 2.0524 | 0.9917 | 0.0466 | 0.0 | referencia |
| `P1-regression-dummy-mean` (baseline) | — | — | 1.6672 | 2.7013 | 1.6381 | 0.0291 | 0.2 | peor que el estacional |

El prefijo de los runs de la búsqueda es `P1-regression-`.

### 4.2. P2 — presión de bicicletas (`bike_pressure`) (padre `P2-search`, run `cf7bf16b…`)

Ordenado por Average Precision (AP). Umbral de decisión 0.5, fijo y declarado antes de entrenar.

| Run | lr | hojas | AP | Precision | Recall | F1 | FP | FN | Ajuste (s) |
|---|---|---|---|---|---|---|---|---|---|
| `…search-lr0.2-leaf63` | 0.2 | 63 | **0.3242** | 0.1478 | 0.8410 | 0.2515 | 106,280 | 3,487 | 16.2 |
| `…search-lr0.1-leaf63` | 0.1 | 63 | 0.3219 | 0.1445 | 0.8522 | 0.2471 | 110,601 | 3,240 | 17.1 |
| `…search-lr0.2-leaf31` | 0.2 | 31 | 0.3218 | 0.1453 | 0.8511 | 0.2482 | 109,757 | 3,265 | 12.0 |
| `…search-lr0.2-leaf15` | 0.2 | 15 | 0.3186 | 0.1426 | 0.8536 | 0.2443 | 112,565 | 3,210 | 10.9 |
| `…search-lr0.1-leaf31` | 0.1 | 31 | 0.3177 | 0.1417 | 0.8564 | 0.2432 | 113,711 | 3,149 | 15.1 |
| `…search-lr0.1-leaf15` | 0.1 | 15 | 0.3119 | 0.1398 | 0.8576 | 0.2404 | 115,707 | 3,123 | 11.2 |
| `…search-lr0.05-leaf63` | 0.05 | 63 | 0.3103 | 0.1400 | 0.8603 | 0.2408 | 115,911 | 3,062 | 17.9 |
| `…search-lr0.05-leaf31` | 0.05 | 31 | 0.3041 | 0.1368 | 0.8633 | 0.2361 | 119,461 | 2,998 | 13.4 |
| `…search-lr0.05-leaf15` | 0.05 | 15 | 0.2952 | 0.1333 | 0.8668 | 0.2310 | 123,607 | 2,921 | 9.9 |
| `P2-bike-pressure-dummy-prior` (baseline) | — | — | 0.0408 | 0.0000 | 0.0000 | 0.0000 | 0 | 21,925 | 0.1 |

### 4.3. P2 — presión de anclajes (`dock_pressure`)

| Run | lr | hojas | AP | Precision | Recall | F1 | FP | FN | Ajuste (s) |
|---|---|---|---|---|---|---|---|---|---|
| `…search-lr0.2-leaf63` | 0.2 | 63 | **0.3882** | 0.1538 | 0.8490 | 0.2604 | 105,658 | 3,417 | 14.8 |
| `…search-lr0.1-leaf63` | 0.1 | 63 | 0.3849 | 0.1510 | 0.8564 | 0.2568 | 108,894 | 3,248 | 16.5 |
| `…search-lr0.2-leaf31` | 0.2 | 31 | 0.3834 | 0.1503 | 0.8563 | 0.2557 | 109,552 | 3,251 | 17.4 |
| `…search-lr0.1-leaf31` | 0.1 | 31 | 0.3795 | 0.1480 | 0.8608 | 0.2526 | 112,098 | 3,149 | 14.8 |
| `…search-lr0.2-leaf15` | 0.2 | 15 | 0.3781 | 0.1483 | 0.8606 | 0.2530 | 111,841 | 3,154 | 10.8 |
| `…search-lr0.05-leaf63` | 0.05 | 63 | 0.3746 | 0.1460 | 0.8631 | 0.2498 | 114,214 | 3,097 | 18.5 |
| `…search-lr0.1-leaf15` | 0.1 | 15 | 0.3669 | 0.1455 | 0.8615 | 0.2489 | 114,496 | 3,133 | 13.2 |
| `…search-lr0.05-leaf31` | 0.05 | 31 | 0.3640 | 0.1423 | 0.8674 | 0.2444 | 118,310 | 3,000 | 14.8 |
| `…search-lr0.05-leaf15` | 0.05 | 15 | 0.3527 | 0.1390 | 0.8716 | 0.2397 | 122,182 | 2,904 | 10.1 |
| `P2-dock-pressure-dummy-prior` (baseline) | — | — | 0.0421 | 0.0000 | 0.0000 | 0.0000 | 0 | 22,623 | 0.1 |

El prefijo de los runs de la búsqueda de P2 es `P2-bike-pressure-` o `P2-dock-pressure-`.

## 5. Resultados de validation y test de los modelos seleccionados

| Problema | Modelo | Métricas en validation | Test (nov.–dic. 2025) |
|---|---|---|---|
| P1 | `hist_gradient_boosting_poisson`, lr 0.2, 63 hojas (champion) | MAE 0.8596; RMSE 1.5649 | No evaluado |
| P1 | Estacional 168 h (mejor baseline) | MAE 1.0383; RMSE 2.0524 | No evaluado |
| P1 | Media por hora (baseline) | MAE 1.6672; RMSE 2.7013 | No evaluado |
| P2 bicicletas | `hist_gradient_boosting_balanced`, lr 0.2, 63 hojas (champion) | AP 0.3242; Precision 0.1478; Recall 0.8410; F1 0.2515 | No evaluado |
| P2 bicicletas | Dummy prior (baseline) | AP 0.0408; Precision 0; Recall 0; F1 0 | No evaluado |
| P2 anclajes | `hist_gradient_boosting_balanced`, lr 0.2, 63 hojas (champion) | AP 0.3882; Precision 0.1538; Recall 0.8490; F1 0.2604 | No evaluado |
| P2 anclajes | Dummy prior (baseline) | AP 0.0421; Precision 0; Recall 0; F1 0 | No evaluado |
| P3 | K-Means con k=2 (champion) | Silhouette 0.4141 (enero–octubre, sin partición de validation) | No aplica |
| P3 | Baseline por cuantiles de actividad | Silhouette 0.4024 | No aplica |

**Qué no se evaluó y por qué:** al cerrar la entrega 2 ningún modelo se había evaluado en test, por lo que la columna Test de esta tabla quedaba vacía. La evaluación posterior de P1 y P2 está en [reports/test_evaluation.md](../reports/test_evaluation.md); P3 no se evalúa en test.

## 6. Justificación de la selección

El criterio de selección es el que ya usa el registry y se declaró antes de buscar: menor `val_mae` en P1 y mayor `val_average_precision` en P2. Para cada pregunta el mejor candidato fue `lr=0.2` con 63 hojas.

| Criterio | P1 (salidas por estación-hora) | P2 (presión de bicicletas y anclajes) |
|---|---|---|
| Desempeño | MAE 0.8596, 17.2 % menor que el mejor baseline (estacional). | AP 0.3242 y 0.3882, unas 7.9 y 9.2 veces la prevalencia (0.0408 y 0.0421), que es el AP del baseline. |
| Impacto en el negocio | Un error medio inferior a una salida por estación-hora sirve para anticipar la demanda y planear la redistribución. | Con umbral 0.5 el recall es 0.84 y 0.85: se pierde cerca del 16 % de los eventos proxy. La precisión es baja (0.15): se alerta en el 23.2 % y el 23.3 % de las estaciones-hora de validation. Sirve como filtro de priorización, no como alarma automática. |
| Estabilidad | Brecha MAE validation − train de 0.040, la mayor de la rejilla pero pequeña: sin sobreajuste marcado. | El orden por AP es consistente: AP sube al subir `learning_rate` y las hojas en ambos objetivos. |
| Costo computacional | 12.3 s de ajuste, 4.9 s más que la opción más barata. | 16.2 s y 14.8 s de ajuste. El costo no discrimina entre candidatos. |

Matices que conviene declarar en el reporte:

- **Empate práctico en P1.** Los tres mejores candidatos difieren menos de 0.0013 en MAE. `lr0.2-leaf31` tiene el mejor RMSE (1.5551 frente a 1.5649), menor brecha y menor tiempo (10.9 s). La regla predeclarada (MAE) elige `lr0.2-leaf63`. Con una sola semilla y una sola partición temporal no hay intervalos de confianza para afirmar que esa diferencia sea significativa.
- **La rejilla no se agota.** El ganador está en el borde (`learning_rate` 0.2 y 63 hojas, los máximos de la rejilla) en P1 y en los dos objetivos de P2. Podría haber mejoras marginales fuera de ella.
- **P2 es un proxy.** Las etiquetas se derivan de flujos y de una capacidad de 2026-09, no de inventario real. El umbral 0.5 no se calibró; su elección operativa es una decisión de negocio pendiente.

## 7. Producción con el alias `champion`

| Modelo registrado | Versión | Run origen | Métrica de selección |
|---|---|---|---|
| `mibici-p1-departures` | 1 | `P1-regression-search-lr0.2-leaf63` | `val_mae` 0.859623 |
| `mibici-p2-bike-pressure` | 1 | `P2-bike-pressure-search-lr0.2-leaf63` | `val_average_precision` 0.324210 |
| `mibici-p2-dock-pressure` | 1 | `P2-dock-pressure-search-lr0.2-leaf63` | `val_average_precision` 0.388196 |
| `mibici-p3-station-clusters` | 1 | `P3-clustering-kmeans-k2` | `silhouette_score` 0.414137 |

Cada versión guarda en tags la pregunta, el modelo, la métrica y su valor, la versión de los datos (`train_validation_data_version`) y el commit de Git; además enlaza al run de origen con sus parámetros, artefactos y manifiestos. Los cuatro modelos se cargaron con `mlflow.pyfunc.load_model("models:/<nombre>@champion")` y se les pidió una predicción con su ejemplo de entrada guardado.

Esquema de producción:

- Los consumidores cargan siempre `models:/<nombre>@champion`, nunca un número de versión. Cambiar de modelo no exige tocar el código que lo usa.
- `register` crea una versión nueva por cada run mejor distinto y **solo mueve el alias si la métrica mejora estrictamente** la del champion actual; si no, la versión queda registrada sin alias. Volver a correrlo sin entrenar no duplica versiones.
- Para revertir: `MlflowClient().set_registered_model_alias(nombre, "champion", versión_anterior)`.
- Se usan alias porque las etapas (`Staging`/`Production`) están obsoletas en MLflow 3.

## 8. Verificación ejecutada

| Verificación | Resultado |
|---|---|
| Suite completa de `pytest` | 112 pruebas ejecutadas, 112 aprobadas, 0 fallidas, en 9 min 3 s (antes de los cambios: 108 aprobadas; se agregaron 4: búsqueda de P1, búsqueda de P2, parseo de la rejilla y una de regresión del registry) |
| `ruff check` y `ruff format --check` | Sin hallazgos; 16 archivos con formato correcto (antes fallaba `mibici/dataset.py`) |
| Manifiesto de datos | SHA-256, tamaños y URLs de los 12 CSV coinciden con `references/mibici_inventory_2025.csv`; no se descargó nada |
| `mlflow run . -e ingest` | Run `FINISHED`; 4,532,032 viajes; tag `data_version` igual al del inventario; huella del dataset horario igual antes y después |
| Jerarquía en la base de tracking | 44 runs `FINISHED` en `mibici-equipo28`: `P1-search` con 11 hijos, `P2-search` con 20 hijos y `P3-clustering` con 8 hijos |
| Candidatos y baselines (31 runs) | Todos con los tags requeridos, `manifests/`, `config/`, `evaluation/`, un modelo con firma y ejemplo de entrada, y huella de datos de 64 caracteres |
| Model Registry | 4 modelos, cada uno con alias `champion` en la versión 1; los 4 se cargan por `@champion` |

### Fallos encontrados y su corrección

| Fallo | Causa | Corrección |
|---|---|---|
| `mlflow run . -e ingest` fallaba con «unexpected extra argument» | `manifest_path` tenía tipo `path`; MLflow lo vuelve ruta absoluta y en Windows la ruta del proyecto tiene un espacio (`MLOps Proyecto`) | `MLproject`: `manifest_path` pasó a tipo `string` (relativo al proyecto). Mismo valor por defecto. |
| `UnicodeEncodeError` de `mlflow run` al redirigir o canalizar la salida en PowerShell | MLflow imprime un emoji y la consola usa `cp1252` | Definir `PYTHONUTF8=1` antes de ejecutar (documentado en la sección 2) |
| La primera búsqueda de P1 terminó con `MemoryError` (225 MiB) | Presión de memoria transitoria del equipo; no se repitió | Se eliminó ese run fallido y se volvió a ejecutar sin cambios |
| `ruff format --check` fallaba en `mibici/dataset.py` | Formato | `ruff format`; solo cambia espacios y saltos de línea |

Observación de rendimiento: cada prueba nueva de búsqueda tarda unos 3 minutos porque MLflow infiere los requisitos de pip de cada modelo en un subproceso. Una corrida con pytest tardó más de una hora y se atribuye a una suspensión del equipo; al repetir la prueba sola tardó 171 s.

## 9. Capturas que debes tomar en la UI de MLflow (http://127.0.0.1:5000)

1. Lista de runs del experimento `mibici-equipo28` con los padres expandidos (`P1-search`, `P2-search`, `P3-clustering`) para mostrar los nested runs.
2. `P1-search` expandido: 11 hijos con las columnas `val_mae`, `val_rmse`, `fit_seconds`, `learning_rate` y `max_leaf_nodes`.
3. Comparación de runs de `P1-search`: selecciona los 9 candidatos y el baseline estacional y pulsa **Compare**; captura el gráfico de dispersión o de coordenadas paralelas.
4. Vista de gráficas del experimento con `val_mae` por run.
5. `P2-search` con el filtro `tags.objective = "bike_pressure"` y otra captura con `dock_pressure`, ordenadas por `val_average_precision`.
6. Run ganador de P1 (`P1-regression-search-lr0.2-leaf63`), pestaña **Overview**: parámetros, métricas y tags (`question`, `model_family`, `stage_of_work`, commit de Git, `train_validation_data_version`).
7. Mismo run, pestaña **Artifacts**: `evaluation/hourly_profile.png`, `manifests/`, `config/run_config.json` y `model/` con la firma y el ejemplo de entrada.
8. Run ganador de P2 (`bike_pressure` y `dock_pressure`): `evaluation/confusion_matrix.png` y `evaluation/precision_recall.png`.
9. Run padre `P1-search` (o `P2-search`): tags `selected_run_*` y el artefacto `reports/selection_summary.json`.
10. Pestaña **Datasets** (entradas) de un candidato, con `station_hourly_2025_train_validation`.
11. Pestaña **Models**: los 4 modelos registrados con su versión y el alias `champion`.
12. Página de la versión 1 de `mibici-p1-departures`: alias, descripción, tags y enlace al run de origen.
13. Run de `ingest`: tag `data_version`, métrica `trips_total` y artefactos `manifests/` y `reports/`.
14. Terminal: el mensaje `=== Run (ID '…') succeeded ===` de `search` y la salida de `pytest` en verde.

## 10. Pendientes que requieren intervención manual

- **Commit y linaje exacto.** Los runs registraron el commit `cb6461e` (la base de la rama), porque el código nuevo no estaba confirmado. Para que el commit del modelo sea el del Pull Request: confirma los cambios y vuelve a correr `search` (P1 y P2) y `register`. Git no tiene `user.name` ni `user.email` configurados en este equipo, por lo que no se hicieron commits.
- **Push y Pull Request.** No se hizo push ni merge.
- **Evaluación en test** de los modelos seleccionados, si el equipo la autoriza (ver sección 5).
- **Capturas** de la sección 9.
- Las notas del handoff `references/p1_p2_mlflow_handoff.md` que dicen que los runs «todavía no son formales» están desactualizadas; no se modificaron.
