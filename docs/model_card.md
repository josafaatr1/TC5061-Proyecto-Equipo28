# Model Card — Clasificadores de presión P2 de MiBici (Equipo 28)

Ficha del modelo que se despliega en la API: los dos clasificadores de presión operacional de P2. Resume su propósito, sus datos, su evaluación, su monitoreo y sus limitaciones. Cada cifra proviene de un documento del repositorio, enlazado en su sección; esta ficha no introduce resultados nuevos.

| Campo | Valor |
|---|---|
| Modelos registrados | `mibici-p2-bike-pressure` (falta de bicicletas) y `mibici-p2-dock-pressure` (falta de anclajes) |
| Versión en producción | v1 de ambos, alias `champion` |
| Versión candidata | `mibici-p2-bike-pressure` v2 (enero-octubre), registrada **sin** alias |
| Tipo | Clasificación binaria por estación-hora |
| Algoritmo | `HistGradientBoostingClassifier` de scikit-learn 1.9.1, `class_weight="balanced"` |
| Seguimiento y registro | MLflow 3.17.0, experimento `mibici-equipo28` |
| Servicio | FastAPI en contenedor (sección 14 del [README](../README.md)), orquestado con Docker Compose |
| Responsables | Equipo 28, TC5061 MLOps, Tecnológico de Monterrey |
| Fecha de la ficha | 2026-11-26 |

Los modelos P1 (regresión de salidas) y P3 (clustering de estaciones) también están en el Model Registry con alias `champion`, pero no se sirven en la API; sus resultados están en [test_evaluation.md](../reports/test_evaluation.md) y [entrega2_mlflow.md](entrega2_mlflow.md).

## 1. Uso previsto

**Propósito.** Priorizar qué estaciones conviene revisar o rebalancear en las próximas tres horas porque podrían quedarse sin bicicletas o sin anclajes. Su usuario previsto es un analista de operación de MiBici que trabaja con una lista ordenada de estaciones.

**Usos fuera de alcance.**

- Alarma automática o decisión sin revisión humana: con el umbral 0.5 la precisión es de 0.12 a 0.14 (sección 4).
- Estimar el inventario real de bicicletas o de anclajes: las etiquetas son proxies de flujo (sección 3).
- Aplicarlo a otro sistema de bicicletas, a otra ciudad o a estaciones nuevas sin historia.
- Cualquier decisión sobre personas: el modelo opera con agregados por estación y no con usuarios.

## 2. Entradas y salidas

La API recibe los 15 predictores históricos de una estación-hora y devuelve una clasificación por objetivo. El consumidor construye los predictores; la API no los calcula a partir de una estación y una fecha. Contrato completo: sección 14.4 del README y [p2-request.json](examples/p2-request.json).

| Grupo | Variables |
|---|---|
| Calendario (4) | `hour_of_day`, `day_of_week`, `month`, `is_weekend` |
| Rezagos (9) | `departures`, `arrivals` y `net_flow` con rezagos de 1 h, 24 h y 168 h |
| Medias móviles (2) | `departures_rolling_mean_24h`, `departures_rolling_mean_168h` |

Todos los predictores usan solo información anterior a la hora consultada.

## 3. Datos y etiquetas

**Fuente.** Datos abiertos de viajes de MiBici de 2025: 4,532,032 viajes y 371 estaciones con actividad, agregados por estación y hora. Durante la estandarización se excluyen `Usuario_Id`, `Genero` y `Año_de_nacimiento` (sección 3.3 del README). La capacidad de cada estación sale del catálogo de septiembre de 2026 (`nomenclatura_2026_09.csv`, SHA-256 `206e45fe…`).

**Etiquetas proxy.** Para una estación y una hora `t` se acumula el flujo neto (llegadas menos salidas) en `t`, `t+1` y `t+2`. La presión de bicicletas es el mayor déficit acumulado y la de anclajes el mayor excedente acumulado, ambos divididos entre la capacidad de la estación. Una ventana es positiva si su presión es mayor o igual que el percentil 90 de las presiones positivas de train: 0.368421 en ambos objetivos. Ese umbral se congeló en train y se aplica sin cambios a validation y test. Las ventanas incompletas o que cruzan particiones se excluyen.

| Partición (2025) | Ventanas | Uso |
|---|---:|---|
| Train, enero-agosto | 2,137,182 | Entrenamiento de v1 y referencia del monitoreo |
| Validation, septiembre-octubre | 536,904 | Selección de hiperparámetros y del `champion` |
| Test, noviembre-diciembre | 536,245 | Evaluación final, drift y comparación de versiones |

La prevalencia de positivos es baja: en test es 3.09 % en bicicletas y 3.43 % en anclajes.

## 4. Entrenamiento y evaluación

**Configuración de v1.** Búsqueda de 9 combinaciones de HistGradientBoosting en validation; ganó `learning_rate=0.2`, `max_leaf_nodes=63`, 100 iteraciones, sin early stopping y semilla 42. El criterio de selección es `val_average_precision`. Umbral de decisión: 0.5, no calibrado.

**Resultados en test** (noviembre-diciembre, evaluado una sola vez y sin reajustar; detalle en [test_evaluation.md](../reports/test_evaluation.md)):

| Objetivo | Conjunto | AP | Precision | Recall | F1 | AP del baseline (prevalencia) |
|---|---|---:|---:|---:|---:|---:|
| Bicicletas | Validation | 0.3242 | 0.1478 | 0.8410 | 0.2515 | 0.0408 |
| Bicicletas | **Test** | **0.2624** | 0.1185 | 0.8175 | 0.2070 | 0.0309 |
| Anclajes | Validation | 0.3882 | 0.1538 | 0.8490 | 0.2604 | 0.0421 |
| Anclajes | **Test** | **0.3646** | 0.1355 | 0.8412 | 0.2334 | 0.0343 |

Matrices de confusión en test, `[[TN, FP], [FN, TP]]`: bicicletas `[[418,971, 100,714], [3,023, 13,537]]`; anclajes `[[419,279, 98,596], [2,917, 15,453]]`.

**Lectura.** El modelo ordena bien el riesgo: su AP es 8.5 veces (bicicletas) y 10.6 veces (anclajes) la del baseline. Detecta más de 80 % de los eventos, pero con el umbral 0.5 marca alrededor del 21 % de las estaciones-hora, por lo que la mayoría de sus alertas son falsas. Por eso sirve para priorizar revisiones y no como alarma automática.

## 5. Registro, linaje y estado de producción

Las versiones v1 se registraron con el entry point `register` (sección 7.7 del README). El alias `champion` solo se mueve a una versión que mejora la métrica del `champion` vigente, y la API consume el modelo por alias, no por número de versión. Cada versión registrada conserva su run de origen y los tags de métrica, versión de datos y commit.

Linaje de los resultados reportados en esta ficha y en la comparación de versiones:

| Modelo | Versión | Estado | Run de origen | Datos de entrenamiento |
|---|---|---|---|---|
| `mibici-p2-bike-pressure` | 1 | `champion` | `d306fe5f7f34496a94f8cc3d0db2f62b` | Enero-agosto |
| `mibici-p2-bike-pressure` | 2 | Candidato, sin alias | `d2df45da747c4241a8ff657417aa8332` | Enero-octubre |
| `mibici-p2-dock-pressure` | 1 | `champion` | `81f93c6f160e4f44afe3db913eac5030` | Enero-agosto |

Huella del conjunto horario completo: `ad5b310c…` (SHA-256 completo en [test_evaluation.md](../reports/test_evaluation.md)). Las evaluaciones verificaron que coincidieran la huella de datos, el SHA-256 del catálogo y el de las etiquetas antes de calcular métricas.

El servidor MLflow es local con SQLite y no se versiona en Git. Cada integrante que reproduce el flujo obtiene **run IDs distintos** con la misma receta; los identificadores de esta tabla son los del entorno donde se generaron la evaluación en test, el reporte de drift y la comparación de versiones.

## 6. Monitoreo

**Drift con Evidently** ([interpretación](drift_interpretation.md), [reporte HTML](../reports/evidently/data_drift_report.html), [resumen JSON](../reports/evidently/drift_summary.json)). Se compararon train (reference) contra noviembre-diciembre (current) en los 15 predictores y las dos etiquetas. Resultado: drift en 3 de 17 variables.

- `month` (Jensen-Shannon 0.83): current contiene meses que el modelo nunca vio en train. Es drift estructural por estacionalidad.
- `target_bike_pressure` (4.34 % → 3.09 %) y `target_dock_pressure` (4.10 % → 3.43 %): cambio de prevalencia, que explica en parte la caída de precision y AP.
- Ninguna de las 11 variables de flujo supera el umbral de Wasserstein normalizado de 0.1.

El reporte se regeneró el 2026-11-26 con `python -m mibici.monitoring.drift_report --catalog-path data/external/nomenclatura_2026_09.csv` y produjo el mismo `drift_summary.json`; los HTML solo difirieron en identificadores internos de Evidently.

**Comparación de versiones (A/B offline)** ([resumen](ab_testing_summary.md), [CSV](../reports/ab_testing/p2_bike_registered_comparison.csv)). v1 y v2 de bicicletas se evaluaron sobre las mismas 536,245 ventanas de test:

| Métrica | v1 `champion` | v2 candidato | Cambio relativo |
|---|---:|---:|---:|
| Average Precision | 0.2624 | 0.2774 | +5.70 % |
| Precision | 0.1185 | 0.1174 | −0.95 % |
| Recall | 0.8175 | 0.8351 | +2.16 % |
| Falsos positivos | 100,714 | 104,008 | +3.27 % |
| Falsos negativos | 3,023 | 2,731 | −9.66 % |

**Decisión:** mantener v1 como `champion`. v2 ordena mejor y detecta más eventos, pero genera 3,294 alertas falsas adicionales, no tiene intervalos de confianza y noviembre-diciembre ya había sido examinado. Antes de promoverla se requiere un periodo posterior independiente y un costo acordado por alerta falsa. El reentrenamiento continuo es una propuesta con criterios de disparo, no un proceso automatizado ([recomendación](continuous_training_recommendation.md)).

## 7. Costo de operación

Resumen del [análisis FinOps](../reports/finops/finops_report.md), basado en la capacidad medida con Locust y en precios on-demand de AWS en México (Central):

- El Compose actual (API + MLflow en una t3.medium) cuesta $31.90 USD al mes y sostiene 71.4 req/s con p95 ≤ 300 ms.
- Con la demanda real de MiBici (lote por estación o app de usuarios) se usa menos del 1 % de esa capacidad. El costo lo domina la capacidad ociosa, no el cómputo por petición.
- Separar MLflow, usar Graviton (t4g.small) y apagar la instancia en horas sin servicio reduce el costo 66.4 %, a $10.71 USD al mes.
- Un lote horario efímero en Fargate costaría $0.53 USD al mes.

## 8. Limitaciones y riesgos

- **Etiquetas proxy.** Representan flujos netos relativos a la capacidad, no inventario observado. Una alerta no significa que la estación se haya vaciado o llenado.
- **Catálogo retrospectivo.** Las capacidades de septiembre de 2026 se aplican a 2025; las estaciones que cambiaron de capacidad quedan mal normalizadas.
- **Estacionalidad.** Train solo cubre enero-agosto; noviembre y diciembre no aparecen en él y producen el drift en `month`.
- **Evaluación limitada.** Una sola partición temporal y una sola semilla, sin intervalos de confianza. Las ventanas se solapan y dependen de la estación, por lo que la incertidumbre real es mayor que la que sugiere el tamaño de muestra.
- **Umbral no calibrado.** El 0.5 no se eligió con costos operativos. La evaluación usa `probabilidad >= 0.5` y la API `model.predict()`, que trata los empates exactos como negativos; los conteos pueden diferir ligeramente.
- **Estaciones sin historia.** Los rezagos de 168 h no existen para estaciones nuevas o con huecos, y el desempeño con esos faltantes no se evaluó por separado.
- **Comparación de versiones offline.** La comparación es retrospectiva y pareada; no se asignó tráfico real a cada versión.

## 9. Consideraciones éticas y de gobernanza

- **Privacidad.** El modelo no usa datos personales: se eliminan los identificadores y atributos demográficos y se trabaja con agregados por estación-hora. Los datos y modelos se guardan localmente, fuera de Git.
- **Equidad territorial.** No se evaluó el desempeño por zona ni por nivel de uso de las estaciones. Usar el modelo como único criterio podría concentrar el rebalanceo en ciertas zonas; se recomienda medir el desempeño por zona antes de operar.
- **Supervisión humana.** Promover una versión o cambiar el umbral requiere aprobación humana. v1 se conserva para rollback.
- **Trazabilidad.** Cada resultado está ligado a un run de MLflow, una huella de datos y un commit; las pruebas automáticas (`pytest`) y el CI de GitHub Actions validan el código en cada cambio.

## 10. Reproducción

Los modelos de esta ficha se entrenaron con el catálogo de septiembre de 2026 (`nomenclatura_2026_09.csv`). En `main`, el catálogo predeterminado de `mibici/modeling/p2_catalog.py` es el de octubre (`nomenclatura_2026_10.csv`) y los entry points `search`, `p2_train` y `evaluate` no permiten cambiarlo: con él, la verificación de etiquetas falla o, si se reconstruyen, cambian las etiquetas y las métricas. Por eso los pasos de P2 y la evaluación llaman a las funciones de Python con `catalog_path` explícito.

**1. Entrenar y registrar el `champion`.** `register` solo compara los hijos del entrenamiento más reciente de cada pregunta. Si el último entrenamiento fue `train`, `p2_train` o `workflow`, registra la configuración por defecto (`learning_rate=0.1`, `max_leaf_nodes=31`, AP de validación 0.3177 en bicicletas y 0.3795 en anclajes), que no es el modelo de esta ficha. La búsqueda debe ser el último entrenamiento antes de registrar:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000 MLFLOW_EXPERIMENT_NAME=mibici-equipo28
python - <<'PY'
from pathlib import Path
from mibici.modeling.mlflow_runs import parse_search_grid, run_p1, run_p2, run_p3

catalog = Path("data/external/nomenclatura_2026_09.csv")
grid = parse_search_grid("0.05,0.1,0.2", "15,31,63")
run_p1(search_grid=grid)                       # búsqueda P1 (no usa el catálogo)
run_p2(search_grid=grid, catalog_path=catalog)  # búsqueda P2 con el catálogo de septiembre
run_p3(catalog_path=catalog)                   # clustering P3
PY
mlflow run . -e register --env-manager local
```

Después, en la pestaña **Models** de MLflow, verificar que la versión con alias `champion` de cada modelo P2 tenga `learning_rate=0.2` y `max_leaf_nodes=63`, y `val_average_precision` de 0.3242 (bicicletas) y 0.3882 (anclajes). Los run IDs serán distintos a los de la sección 5, porque MLflow genera un ID nuevo en cada ejecución; las métricas sí deben coincidir, ya que la semilla está fija.

**2. Evaluar, monitorear y estimar costos.** Con ese `champion` registrado y los Parquets procesados, desde la raíz del repositorio:

```bash
python -c "from pathlib import Path; from mibici.modeling.evaluate import run_evaluation; run_evaluation(catalog_path=Path('data/external/nomenclatura_2026_09.csv'))"  # evaluación en test
python -m mibici.monitoring.drift_report --catalog-path data/external/nomenclatura_2026_09.csv  # drift
python -m mibici.modeling.retrain_p2_bike compare --version 2  # comparación v1/v2
python -m mibici.finops                                       # FinOps
```

Estos pasos se verificaron el 2026-11-26 en un servidor MLflow nuevo. Los cuatro modelos quedaron como v1 `champion`, con las mismas métricas de validación (P1 MAE 0.8596; P2 AP 0.3242 y 0.3882; P3 `kmeans_k2`, silhouette 0.4141), y la evaluación reprodujo exactamente las métricas y matrices de confusión de test de la sección 4.

La comparación de versiones requiere además la v2 de bicicletas (`python -m mibici.modeling.retrain_p2_bike train`). `retrain_p2_bike` ya usa el catálogo de septiembre por defecto. Pruebas del monitoreo, FinOps y modelado: `pytest tests/monitoring tests/test_finops.py tests/modeling` (108 aprobadas el 2026-11-26).
