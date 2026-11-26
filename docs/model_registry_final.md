# Modelo y Model Registry — Proyecto MiBici

## 1. Objetivo

Documentar la selección, el registro, el versionado y la trazabilidad de los modelos del proyecto MiBici mediante MLflow Model Registry.

Esta documentación corresponde a la responsabilidad del Integrante 2: Modelo y Model Registry.

## 2. Entorno de MLflow

El registro y seguimiento de los modelos se realizaron utilizando MLflow.

- Versión de MLflow: 3.17.0.
- Tracking URI: `http://127.0.0.1:5000`.
- Backend de almacenamiento: SQLite (`mlflow.db`).
- Experimento principal: `mibici-equipo28`.
- Experiment ID: `1`.

El servidor utilizado es local. La base de datos del Registry y los artefactos almacenados no se publican automáticamente mediante GitHub.

## 3. Modelos registrados

Se verificaron cuatro modelos en MLflow Model Registry. Cada uno cuenta con una versión registrada y el alias `champion`.

| Problema | Modelo registrado | Versión | Métrica de selección | Resultado |
|---|---|---:|---|---:|
| P1 — Predicción de salidas | `mibici-p1-departures` | 1 | `val_mae` (menor es mejor) | 0.864087 |
| P2 — Presión de bicicletas | `mibici-p2-bike-pressure` | 1 | `val_average_precision` (mayor es mejor) | 0.317719 |
| P2 — Presión de anclajes | `mibici-p2-dock-pressure` | 1 | `val_average_precision` (mayor es mejor) | 0.379491 |
| P3 — Agrupación de estaciones | `mibici-p3-station-clusters` | 1 | `silhouette_score` (mayor es mejor) | 0.414137 |

Los resultados corresponden a las ejecuciones verificadas en el servidor local de MLflow. Para P1 y P2 se utilizaron métricas de validación; para P3 se utilizó silhouette score.

## 4. Linaje y trazabilidad

Cada versión registrada conserva una referencia a su ejecución de origen. Esto permite consultar los parámetros, las métricas, las etiquetas y los artefactos asociados al entrenamiento.

### 4.1. P1 — Predicción de salidas

- Modelo registrado: `mibici-p1-departures`.
- Versión: `1`.
- Alias: `champion`.
- Run ID: `a15e459118004312b1cf500e7c5c54eb`.
- Modelo seleccionado: `hist_gradient_boosting_poisson`.
- MAE de validación: `0.864087`.
- RMSE de validación: `1.570014`.

El run contiene parámetros del entrenamiento, métricas de evaluación y artefactos como la configuración de ejecución, manifiestos de datos y archivos de evaluación.

### 4.2. P2 — Presión de bicicletas

- Modelo registrado: `mibici-p2-bike-pressure`.
- Versión: `1`.
- Alias: `champion`.
- Run ID: `3cecf9c99fdc477abce610da3a04af13`.
- Métrica `val_average_precision`: `0.317719`.

El run conserva métricas de clasificación, parámetros, etiquetas de procedencia y artefactos de evaluación.

### 4.3. P2 — Presión de anclajes

- Modelo registrado: `mibici-p2-dock-pressure`.
- Versión: `1`.
- Alias: `champion`.
- Run ID: `1b89a7da4676463e8f9421f5f09fdf51`.
- Métrica `val_average_precision`: `0.379491`.

El run conserva los parámetros y métricas de clasificación, así como los artefactos de evaluación y manifiestos asociados.

### 4.4. P3 — Agrupación de estaciones

- Modelo registrado: `mibici-p3-station-clusters`.
- Versión: `1`.
- Alias: `champion`.
- Run ID: `1f0262d70f6242be8e54ac4f857f0957`.
- Modelo seleccionado: `kmeans_k2`.
- Silhouette score: `0.414137`.

El run contiene los parámetros del algoritmo, las métricas de agrupación y los artefactos asociados al resultado del clustering.

## 5. Alias de producción

Los cuatro modelos tienen definido el alias `champion` en su versión 1.

Este alias identifica la versión seleccionada para ser consumida por los componentes que requieren realizar inferencias.

Se utilizan alias para evitar depender directamente de un número de versión. De esta manera, cuando una nueva versión sea seleccionada, el alias puede apuntar a ella sin modificar el identificador utilizado por los consumidores.

## 6. Recuperación de los modelos

Los modelos pueden recuperarse mediante la API de MLflow usando su nombre registrado y el alias `champion`.

Ejemplo para P1:

```python
import mlflow

model = mlflow.pyfunc.load_model(
    "models:/mibici-p1-departures@champion"
)
```

Para recuperar los demás modelos se utiliza el mismo procedimiento, sustituyendo el nombre:

```text
models:/mibici-p2-bike-pressure@champion
models:/mibici-p2-dock-pressure@champion
models:/mibici-p3-station-clusters@champion
```

Es necesario que el cliente tenga configurado el mismo Tracking URI y acceso al almacenamiento de artefactos correspondiente.

## 7. Procedimiento de registro

El registro de modelos se realiza mediante el módulo:

```powershell
python -m mibici.modeling.registry main
```

Este procedimiento consulta las ejecuciones disponibles, identifica los mejores modelos de acuerdo con las métricas configuradas y actualiza el Model Registry.

La lógica de registro contempla la comparación con el modelo que actualmente tiene el alias `champion`, evitando promover una versión que no supere el resultado vigente.

## 8. Evidencia en MLflow

Las capturas de pantalla correspondientes a esta verificación se conservarán en:

`docs/evidence/model_registry/`

Se consideran relevantes las siguientes evidencias:

1. Lista de los cuatro modelos registrados y sus versiones.
2. Alias `champion` de cada modelo.
3. Detalle de las versiones registradas, incluyendo sus ejecuciones de origen.
4. Métricas y parámetros de los runs de entrenamiento.
5. Artefactos disponibles en cada ejecución.

## 9. Conclusión

La revisión confirma que los cuatro modelos del proyecto se encuentran registrados y versionados en MLflow Model Registry, cada uno con una versión 1 y el alias `champion`.

Además, las versiones conservan referencias a sus ejecuciones de origen, lo que permite consultar sus métricas, parámetros y artefactos asociados.

Con esto se documentan el registro, la trazabilidad y el mecanismo de recuperación de los modelos correspondientes a P1, P2 y P3.
   