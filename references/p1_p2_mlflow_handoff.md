# Handoff técnico P1/P2 para integración con MLflow

**Alcance:** documentar las interfaces y resultados existentes para coordinar la
integración de los modelos. Este documento no integra MLflow, no modifica el
pipeline de entrenamiento y no evalúa el conjunto de test.

## 1. Objetivos y formulación

| Problema | Formulación implementada |
|---|---|
| P1 | Regresión por estación-hora para pronosticar `departures` en la hora objetivo. |
| P2 bicicletas | Clasificación binaria de presión proxy por desequilibrio de salidas durante las tres horas futuras. |
| P2 anclajes | Clasificación binaria de presión proxy por desequilibrio de llegadas durante las tres horas futuras. |

Las etiquetas P2 no representan inventario real ni confirman que una estación
se haya quedado sin bicicletas o anclajes. Los datos de viajes no contienen
inventarios históricos.

## 2. Arquitectura y dependencias

1. Los CSV mensuales oficiales de viajes de 2025 se preparan mediante
   [`dataset.py`](../mibici/dataset.py) y producen
   `data/processed/station_hourly_2025_complete.parquet`.
2. P1 consume ese Parquet y genera las características mediante
   [`features.py`](../mibici/features.py), invocado desde
   [`train.py`](../mibici/modeling/train.py). El comando independiente de
   `features.py` guarda `features_p1_2025.parquet`, pero no es un prerrequisito
   de `train.py`.
3. P2 obtiene capacidades del catálogo con
   [`p2_catalog.py`](../mibici/modeling/p2_catalog.py), construye ventanas,
   características y etiquetas con
   [`p2_targets.py`](../mibici/modeling/p2_targets.py), y las vuelve a verificar
   antes de entrenar desde [`p2_train.py`](../mibici/modeling/p2_train.py).
4. Las dos tareas P2 se entrenan y evalúan independientemente, usando las
   mismas filas de validación.

La configuración común está en [`config.py`](../mibici/config.py): semilla 42,
periodo 2025 y rutas de datos. Las dependencias están fijadas en
`requirements.txt`, `requirements-lock.txt` y `python_env.yaml`; el ambiente
MLflow declara Python 3.14.4, MLflow 3.17.0, pandas 3.0.6, NumPy 2.5.4,
PyArrow 25.0.1 y scikit-learn 1.9.1.

## 3. Fuentes y procedencia de datos

### Viajes y datos horarios

La fuente declarada es el portal de
[Datos abiertos de MiBici](https://www.mibici.net/es/datos-abiertos/). El
repositorio contiene el manifiesto `references/mibici_sources_2025.csv` y el
inventario mensual con hashes `references/mibici_inventory_2025.csv`. El
Parquet horario usado por P1 y P2 es
`data/processed/station_hourly_2025_complete.parquet`.

El CLI de P1 no incorpora una huella de datos en su reporte actual. Antes de
registrar los runs, debe calcularse y guardarse una huella reproducible del
conjunto acotado a train/validation; no se reporta aquí un hash P1 que no esté
disponible en los artefactos existentes.

### Catálogo de estaciones P2

| Campo | Valor |
|---|---|
| Catálogo | `nomenclatura_2026_09.csv` |
| Fuente oficial | [Datos abiertos MiBici](https://www.mibici.net/es/datos-abiertos/) |
| URL directa registrada | `https://www.mibici.net/site/assets/files/1118/nomenclatura_2026_09.csv` |
| Versión | Septiembre de 2026 |
| SHA-256 | `206e45fe633922b634d858142fa21e6fd9f4c421b6435c02980ef290cc2df90d` |
| Cobertura reportada | 370 de 370 estaciones utilizadas entre enero y octubre de 2025 tienen capacidad positiva y única. |

El catálogo contiene 484 estaciones. Cuatro capacidades son cero (IDs 99, 109,
116 y 138), y ninguna corresponde a las 370 estaciones utilizadas en el
periodo auditado. Las capacidades son retrospectivas y no están validadas para
2025; la cobertura no elimina esta limitación temporal.

### Artefacto de etiquetas P2

`data/processed/p2_targets_2025_train_validation.parquet` contiene solo
ventanas train y validation. Su SHA-256 registrado es
`47009f1117fc05eedeeae3c1e236f3d05007b45dd3fa09b220d4929f791b4541`.
El reporte de entrenamiento confirma `labels_reproduced_exactly=True` y
registra la versión/hash del catálogo y los umbrales aprendidos. No incluye un
hash explícito del Parquet horario de origen; conviene agregar una huella
acotada a los meses usados antes de cerrar la trazabilidad en MLflow.

## 4. Características, particiones y control de fuga

### P1

- Objetivo: `departures` de la hora objetivo.
- Predictores: hora del día, día de semana, mes, fin de semana; rezagos
  estación-específicos de salidas, llegadas y flujo neto a 1, 24 y 168 horas;
  medias móviles históricas de salidas a 24 y 168 horas.
- `station_id`, timestamp y etiqueta no entran en la matriz de predictores.
  `build_features` conserva solamente claves, calendario, rezagos y medias; no
  conserva los conteos de la hora objetivo.
- Cada rezago se busca por hora real y estación; las discontinuidades no se
  rellenan como si fueran observaciones.
- La predicción se define antes del inicio de la hora objetivo. Las medias
  móviles excluyen esa hora.

### P2

- Predictores: los cuatro atributos de calendario, los nueve rezagos de
  salidas/llegadas/flujo neto a 1, 24 y 168 horas, y las dos medias móviles
  históricas de salidas a 24 y 168 horas. `station_id`, `hour`, `partition`,
  objetivos, presiones y ratios de presión se excluyen.
- Para una hora ancla `t`, se usan flujos observados en `t`, `t+1` y `t+2`,
  con `net_flow = arrivals - departures`. Si `S_k` es el flujo neto acumulado
  hasta la hora `t+k-1`, las presiones son
  `bike_pressure = max(0, -min(S1, S2, S3))` y
  `dock_pressure = max(0, max(S1, S2, S3))`. Ambas se dividen entre la
  capacidad positiva de la estación.
- Los umbrales se ajustan separadamente al percentil 90 de las presiones
  normalizadas positivas de ventanas completas de train; se congelan para
  validation. Una etiqueta es positiva cuando el ratio es mayor o igual al
  umbral.
- Umbrales registrados: bicicletas `0.3684210526`; anclajes `0.3684210526`.
  La capacidad del catálogo 2026 se usa como aproximación, no como inventario
  histórico.

| Partición | Intervalo | P1 | P2 |
|---|---|---:|---:|
| Train | 2025-01-01 a 2025-09-01 (fin exclusivo) | 2,137,920 filas | 2,137,182 ventanas |
| Validation | 2025-09-01 a 2025-11-01 (fin exclusivo) | 537,640 filas originales; 537,472 evaluables en la cohorte común | 536,904 ventanas |
| Test reservado | 2025-11-01 a 2026-01-01 (fin exclusivo) | No evaluado | No incluido en el artefacto de etiquetas ni evaluado |

P1 compara sus tres métodos en las mismas 537,472 filas con rezago semanal
disponible. P2 exige ventanas de tres horas exactas, no imputa horas faltantes
y excluye ventanas que atraviesan las fronteras temporales. La CLI de P1
filtra por defecto las filas anteriores al 1 de noviembre; `--evaluate-test`
es una opción explícita que no debe usarse en esta etapa. La CLI de P2 lee las
particiones train y validation del artefacto y rechaza etiquetas de test.

## 5. Modelos e hiperparámetros

### P1

| Método | Definición |
|---|---|
| Dummy | `DummyRegressor(strategy="mean")`, ajustado solo con train. |
| Baseline estacional | Valor de `departures_lag_168h`, misma hora de la semana anterior. |
| Candidato | `HistGradientBoostingRegressor(loss="poisson", learning_rate=0.1, max_iter=100, max_leaf_nodes=31, min_samples_leaf=50, l2_regularization=1.0, early_stopping=False, random_state=42)`. |

Se excluye `station_id` para evitar tratarlo como variable ordinal. No se hace
búsqueda extensa de hiperparámetros.

### P2

| Método | Definición |
|---|---|
| Baseline | `DummyClassifier(strategy="prior")`. |
| Candidato para cada objetivo | `HistGradientBoostingClassifier(loss="log_loss", learning_rate=0.1, max_iter=100, max_leaf_nodes=31, min_samples_leaf=50, l2_regularization=1.0, class_weight="balanced", early_stopping=False, random_state=42)`. |

Los pesos de clase se determinan por el estimador exclusivamente con las
etiquetas de train. El umbral de clasificación es 0.5, predeclarado; no se
calibró con validation. Se evita el early stopping aleatorio interno.

## 6. Resultados reales de validación

### P1

| Modelo | MAE | RMSE | Filas |
|---|---:|---:|---:|
| DummyRegressor | 1.667236 | 2.701261 | 537,472 |
| Estacional 168 h | 1.038283 | 2.052411 | 537,472 |
| HistGradientBoosting Poisson | 0.864087 | 1.570014 | 537,472 |

### P2

En todas las filas, `n=536,904`; no hubo exclusiones adicionales durante la
evaluación. La matriz se presenta como `[[TN, FP], [FN, TP]]`.

| Objetivo | Modelo | AP | Precision | Recall | F1 | Prevalencia positiva | Matriz de confusión |
|---|---|---:|---:|---:|---:|---:|---|
| Bicicletas | Dummy prior | 0.040836 | 0.000000 | 0.000000 | 0.000000 | 0.040836 | `[[514979, 0], [21925, 0]]` |
| Bicicletas | HistGradientBoosting balanceado | 0.317719 | 0.141720 | 0.856374 | 0.243194 | 0.040836 | `[[401268, 113711], [3149, 18776]]` |
| Anclajes | Dummy prior | 0.042136 | 0.000000 | 0.000000 | 0.000000 | 0.042136 | `[[514281, 0], [22623, 0]]` |
| Anclajes | HistGradientBoosting balanceado | 0.379491 | 0.148010 | 0.860805 | 0.252589 | 0.042136 | `[[402183, 112098], [3149, 19474]]` |

El umbral 0.5 hace que el baseline de prior prediga la clase negativa en todas
las filas, dado que la prevalencia es menor que 0.5. En la ejecución reportada,
el ajuste de cada candidato tardó aproximadamente 12.18 s para bicicletas y
11.47 s para anclajes; el proceso completo de P2 registró 32.494 s y un pico
de memoria de 1.747 GiB. No se dispone aquí de una medición verificada de
tiempo/memoria comparable para P1.

Estas métricas ya fueron calculadas en validation, pero todavía no son runs
formales de MLflow. No se han calculado métricas para test.

## 7. Limitaciones metodológicas

- Las etiquetas P2 son proxies de desequilibrio, no eventos observados de
  estación vacía/llena. No hay inventario histórico, fallas operativas ni
  disponibilidad real de anclajes.
- La capacidad proviene de septiembre de 2026 y puede diferir de la capacidad
  de 2025.
- El desbalance de P2 es alto. El candidato mejora AP frente al baseline, pero
  el umbral 0.5 produce muchos falsos positivos; su elección operativa requiere
  una decisión de negocio y no debe ajustarse mirando validation si se quiere
  preservar esta evaluación.
- Ventanas P2 adyacentes se solapan y las observaciones horarias de una
  estación son dependientes temporalmente.
- P1 no modela efectos persistentes de estación porque excluye el identificador.
- Los resultados actuales no guardan todos los datos de procedencia y no se han
  registrado en MLflow.

## 8. Comandos reproducibles comprobados

Los siguientes módulos expusieron el CLI esperado al ejecutar `--help` desde
`.venv\Scripts\python.exe`; no se volvió a entrenar ni evaluar durante esta
auditoría. Ejecutarlos desde la raíz del repositorio con el ambiente
`python_env.yaml` activo:

```powershell
# P1: train + validation; no incluye test por defecto.
python -m mibici.modeling.train

# P2: descarga catálogo oficial y audita cobertura local.
python -m mibici.modeling.p2_catalog

# P2: construye las etiquetas proxy de train y validation.
python -m mibici.modeling.p2_targets

# P2: verifica reproducibilidad y ejecuta train + validation.
python -m mibici.modeling.p2_train
```

Los CLIs permiten sobrescribir las rutas mediante `--input-path`,
`--metrics-path`, `--catalog-path`, `--output-path`, `--label-path` y
`--hourly-path`, según el módulo. `p2_catalog` no tiene parámetros CLI. Los
valores predeterminados apuntan a los artefactos indicados en este documento.
No agregar `--evaluate-test` a la ejecución P1. No existe una opción `--run-id`
en los CLIs P1/P2 actuales.

## 9. Contrato propuesto para runs MLflow

Usar un único experimento, `mibici-equipo28`, con un run por combinación de
problema/objetivo y modelo:

| Run name sugerido | Objetivo |
|---|---|
| `P1-regression-dummy-mean` | Baseline Dummy P1 |
| `P1-regression-seasonal-168h` | Baseline estacional P1 |
| `P1-regression-hist-gradient-boosting-poisson` | Candidato P1 |
| `P2-bike-pressure-dummy-prior` | Baseline de presión proxy de bicicletas |
| `P2-bike-pressure-hist-gradient-boosting-balanced` | Candidato de bicicletas |
| `P2-dock-pressure-dummy-prior` | Baseline de presión proxy de anclajes |
| `P2-dock-pressure-hist-gradient-boosting-balanced` | Candidato de anclajes |

Cada run debe registrar:

- **Parámetros/tags:** pregunta (`P1` o `P2`), objetivo, Equipo 28, clase y
  nombre del modelo, hiperparámetros completos, `random_state=42`, columnas
  predictoras, periodos y tamaños de train/validation.
- **Trazabilidad:** hash y versión del conjunto de datos usado, SHA del commit
  Git, versiones de Python/scikit-learn y, para P2, versión/hash del catálogo,
  hash del artefacto de etiquetas, cuantiles/umbrales proxy y umbral de
  clasificación.
- **Métricas con prefijo `val_`:** P1 `val_mae`, `val_rmse`; P2
  `val_average_precision`, `val_precision`, `val_recall`, `val_f1`,
  `val_positive_prevalence`, `val_n_evaluated`, `val_n_excluded` y celdas de
  confusión `val_tn`, `val_fp`, `val_fn`, `val_tp`.
- **Artefactos agregados:** CSV de comparación; matrices de confusión y
  gráficos/datos de Precision-Recall para P2; reporte de parámetros y
  procedencia. No registrar credenciales, identificadores personales ni datos
  crudos.

Las cifras de validación de la sección 6 existen en salidas locales; todavía
no hay evidencia de que hayan sido registradas formalmente en MLflow.
`p2_train.py` ya produce un JSON con parámetros, procedencia y métricas de
validación. P1 produce un CSV de métricas. Los gráficos no son artefactos
generados actualmente. Para P1, `run_model_comparison` devuelve métricas, no
estimadores ni predicciones; la generación de artefactos que requieran
predicciones debe integrarse mediante una interfaz coordinada, sin duplicar
el entrenamiento.

## 10. Estado de integración y bloqueos

- [`MLproject`](../MLproject) solo declara activos `load` y `workflow`.
  [`workflow.py`](../mibici/workflow.py) ejecuta únicamente `load`; no orquesta
  features ni entrenamiento supervisado. Los ejemplos comentados de `features`
  y `train` usan `--run-id`, opción que no existe en las CLIs actuales. No hay
  entry points activos para P1/P2.
- El `load` activo llama a `dataset.run-pipeline`, que descarga y procesa los
  doce meses de 2025. Por tanto, `mlflow run . -e workflow` toca los datos de
  noviembre-diciembre aunque no calcule métricas de test. No usar ese workflow
  durante esta evaluación con test reservado hasta coordinar una carga limitada
  temporalmente. Esto es distinto de los comandos directos P1/P2 anteriores.
- [`tracking.py`](../mibici/tracking.py) registra el pipeline de datos, no
  experimentos de modelos. Su `summarize_raw_files` y
  `calculate_data_version` recorren los 12 meses de 2025; no deben reutilizarse
  sin cambios para la huella de un experimento que reserva noviembre-diciembre.
  Se necesita acordar una huella acotada a los datos de train/validation.
- Los reportes actuales no registran todos los metadatos requeridos por el
  contrato: falta el hash acotado de entrada P1, el hash explícito del Parquet
  horario fuente de P2 y el commit Git ejecutado. Ningún modelo se serializa
  actualmente.
- `data/**` y `*.joblib` están excluidos de Git. Las dependencias principales
  están fijadas; `pytest` y Ruff también tienen versiones fijadas en los
  manifiestos de desarrollo.
- La integración necesita coordinación con el responsable de
  [`MLproject`](../MLproject), [`workflow.py`](../mibici/workflow.py) y
  [`tracking.py`](../mibici/tracking.py). No se modificaron en esta etapa.
