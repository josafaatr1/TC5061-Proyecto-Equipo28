# Comparación retrospectiva de versiones P2

## Comparación principal: bicicletas v1 frente a v2

| Versión | Entrenamiento | Run de origen | Modelo MLflow |
|---|---|---|---|
| A: v1, `champion` | Enero-agosto: 2,137,182 ventanas | `d306fe5f7f34496a94f8cc3d0db2f62b` | `m-c65129c4cf4d4744b31f026ab2a4c1c1` |
| B: v2, challenger | Enero-octubre: 2,674,086 ventanas | `d2df45da747c4241a8ff657417aa8332` | `m-d632547af1134d33b65648440c2a918b` |

Ambas versiones pertenecen a `mibici-p2-bike-pressure`. B reutiliza los hiperparámetros de A seleccionados previamente con septiembre-octubre: HGB balanceado, learning rate 0.2, 63 hojas, 100 iteraciones y semilla 42. B se ajustó con train y validation unidos, manteniendo las ventanas de cada partición y los umbrales de etiquetas originales de train. No se usó noviembre-diciembre para ajustar B, sus parámetros ni el umbral. Como B incorpora la antigua validation, esa partición ya no es independiente para B.

El run de evaluación `9e2ad353a2204ead83043b2e81426e9a` evaluó ambas versiones en las mismas 536,245 ventanas de noviembre-diciembre, con las mismas etiquetas y umbral 0.5. Las diferencias relativas son `(B-A)/A`; el [CSV definitivo](../reports/ab_testing/p2_bike_registered_comparison.csv) contiene valores completos.

| Métrica | A v1 | B v2 | Cambio absoluto | Cambio relativo |
|---|---:|---:|---:|---:|
| Average Precision | 0.262416 | 0.277382 | +0.014966 | +5.70 % |
| Precision | 0.118485 | 0.117357 | -0.001128 | -0.95 % |
| Recall | 0.817452 | 0.835085 | +0.017633 | +2.16 % |
| F1 | 0.206970 | 0.205793 | -0.001177 | -0.57 % |
| Verdaderos negativos | 418,971 | 415,677 | -3,294 | -0.79 % |
| Falsos positivos | 100,714 | 104,008 | +3,294 | +3.27 % |
| Falsos negativos | 3,023 | 2,731 | -292 | -9.66 % |
| Verdaderos positivos | 13,537 | 13,829 | +292 | +2.16 % |
| Prevalencia | 0.030881 | 0.030881 | 0 | 0 % |

B mejora AP y recall, pero agrega 3,294 falsos positivos y reduce ligeramente precision y F1. La recomendación es **mantener provisionalmente v1**, sin promoción automática, hasta evaluar costos operativos y un periodo independiente; los resultados no establecen significancia estadística. El alias `champion` sigue en v1.

La comparación es offline, pareada y retrospectiva, no un A/B aleatorizado en producción. La receta de B se decidió después de examinar noviembre-diciembre, por lo que ese periodo no es una prueba confirmatoria independiente, aunque no se usó para entrenar, ajustar hiperparámetros ni calibrar B. El catálogo de septiembre de 2026 es retrospectivo y las etiquetas son proxies, no inventario real. No se calcularon intervalos de confianza.

### Trazabilidad y reproducción

La huella del conjunto horario completo es `ad5b310cf014800e6ba0ab16032b74b0ab0909a2a942aea46a20e1646839ec68`. El SHA-256 del catálogo es `206e45fe633922b634d858142fa21e6fd9f4c421b6435c02980ef290cc2df90d` y el de etiquetas es `47009f1117fc05eedeeae3c1e236f3d05007b45dd3fa09b220d4929f791b4541`. El código ejecutado quedó guardado como artefacto `code/retrain_p2_bike.py` del run B, con SHA-256 `348dd8e165c97286f64aac141b1a729b44bb5a2ffc4d0004a6a4376e18316a4a`.

Con el servidor MLflow apuntando a la base y artefactos originales, el catálogo histórico y los Parquets locales:

```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
$env:DO_NOT_TRACK = "1"
.\.venv\Scripts\python.exe -m mibici.modeling.retrain_p2_bike compare --version 2
.\.venv\Scripts\python.exe -m mibici.monitoring.entrega8_figures
```

`compare` registra un nuevo run de evaluación y vuelve a escribir el CSV; no cambia alias. El comando `train` registra una versión y rechaza repetir la misma receta ya registrada. Las bases, datos y modelos están fuera de Git y deben conservarse o exportarse para reproducir los resultados en otra máquina.

La evaluación clasifica como positiva una probabilidad `>= 0.5`; la API usa `model.predict()` y los empates exactos de estos HGB se clasifican como negativos. Revisar esta diferencia antes de extrapolar los conteos a la API.

## Comparación exploratoria anterior

La siguiente comparación es histórica y **no** es la v2 reentrenada ni la tabla principal de esta entrega.

## Pregunta y diseño

Para cada objetivo se compararon dos instancias del mismo `HistGradientBoostingClassifier` balanceado, entrenadas en enero-agosto de 2025 y evaluadas sobre las mismas 536,245 ventanas de noviembre-diciembre de 2025. Se usan las mismas 15 entradas, etiquetas proxy, umbrales de presión de train y umbral de decisión 0.5. Septiembre-octubre se usó para seleccionar el champion, no para entrenar estas instancias.

La variante A es `champion` del registro, versión 1, con tasa 0.2 y 63 hojas. La B es la configuración inicial de la búsqueda, tasa 0.1 y 31 hojas. B es un modelo registrado como artefacto de un run, **no una versión 2 del Model Registry** y no fue reentrenada con meses recientes. Se escogió por ser la configuración inicial, no por el resultado observado en current.

| Objetivo | A: run de origen | B: run de origen |
|---|---|---|
| Bicicletas | `d306fe5f7f34496a94f8cc3d0db2f62b` | `c6bcec6501404d57b0a728bfa3e0f3fd` |
| Anclajes | `81f93c6f160e4f44afe3db913eac5030` | `6c2da2372de840b08e06258acbe07a2b` |

Las cuatro instancias proceden del padre `cf7bf16b9e7449058e69f59c3dfd62ab`. Sus IDs de modelo y hashes están en [p2_comparison.csv](../reports/ab_testing/p2_comparison.csv). La huella del año horario es `ad5b310cf014800e6ba0ab16032b74b0ab0909a2a942aea46a20e1646839ec68`. Las salidas de A reproducen las celdas y métricas de [test_evaluation.md](../reports/test_evaluation.md).

La evaluación clasifica como positiva una probabilidad `>= 0.5`; la API usa `model.predict()` y los empates exactos de estos HGB se clasifican como negativos. La posible diferencia en empates debe revisarse antes de extrapolar los conteos a la API.

## Métricas observadas

| Objetivo | Variante | AP | Precision | Recall | F1 | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|
| Bicicletas | A, champion v1 | 0.2624 | 0.1185 | 0.8175 | 0.2070 | 100,714 | 3,023 |
| Bicicletas | B, alternativa | 0.2609 | 0.1142 | 0.8362 | 0.2009 | 107,432 | 2,712 |
| Anclajes | A, champion v1 | 0.3646 | 0.1355 | 0.8412 | 0.2334 | 98,596 | 2,917 |
| Anclajes | B, alternativa | 0.3613 | 0.1305 | 0.8526 | 0.2263 | 104,378 | 2,708 |

A presenta mayor AP, precision y F1 en ambos objetivos; B recupera más positivos, a costa de más falsos positivos. No se han calculado intervalos de confianza ni definido costos operativos; las diferencias no demuestran superioridad estadística. Mantener A de forma provisional es más prudente que promover B, no una decisión irreversible.

## Límites importantes

Esto es una **comparación offline pareada**, no un experimento A/B aleatorizado en producción. Noviembre-diciembre ya se utilizó para evaluar e interpretar los champion antes de esta comparación, por lo que no es un conjunto de confirmación independiente para promover esta alternativa. Las instancias comparten train y difieren en hiperparámetros; este ejercicio anterior no comprueba que reentrenar con datos recientes mejore el modelo. La v2 reentrenada posterior se presenta en la comparación principal. Para una decisión de promoción se necesita un periodo temporal posterior, aún no examinado, criterios predefinidos y validación por bloques de tiempo/estación.

## Reproducción

Con el entorno del proyecto, el Parquet horario completo, las etiquetas y el catálogo de septiembre de 2026 con su JSON de procedencia, y los artefactos locales de los cuatro runs en `mlartifacts/1`:

```powershell
$env:PYTHONUTF8 = "1"
.\.venv\Scripts\python.exe -m mibici.modeling.ab_comparison
```

El comando verifica hashes del catálogo y etiquetas, procedencia de train/validation, umbrales y asociación run/modelo; reconstruye current una sola vez, evalúa ambos modelos sobre las mismas filas y escribe el CSV. No inicia runs, registra versiones ni modifica alias. Para otra ubicación de artefactos, usar `--artifacts-dir`. Sin los artefactos locales y datos ignorados por Git, la reproducción requiere restaurarlos del almacenamiento MLflow y las fuentes originales.
