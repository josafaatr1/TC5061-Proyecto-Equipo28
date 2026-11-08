# Integración del modelo P1 con MLflow Model Registry

## 1. Objetivo

Establecer el procedimiento para recuperar y utilizar el modelo de predicción de salidas de bicicletas del proyecto MiBici 2025, mediante MLflow Model Registry.

La integración proporciona una interfaz reutilizable para que el servicio REST desarrollado con FastAPI pueda generar predicciones sin incorporar la lógica de entrenamiento.

## 2. Modelo seleccionado

| Propiedad | Valor |
|---|---|
| Nombre registrado | `mibici-p1-departures` |
| Alias | `champion` |
| Versión verificada | `1` |
| Algoritmo | HistGradientBoosting Poisson |
| Problema | P1 — Regresión de salidas por estación-hora |
| MAE de validación | 0.864087 |
| URI de inferencia | `models:/mibici-p1-departures@champion` |

La versión 1 fue registrada y verificada en el entorno local de desarrollo. El alias `champion` permite referenciar el modelo seleccionado sin establecer una versión fija en el código de inferencia.

## 3. Características de entrada

El modelo requiere 15 características previamente calculadas:

| Característica | Tipo MLflow |
|---|---|
| `hour_of_day` | integer |
| `day_of_week` | integer |
| `month` | integer |
| `is_weekend` | boolean |
| `departures_lag_1h` | double |
| `departures_lag_24h` | double |
| `departures_lag_168h` | double |
| `arrivals_lag_1h` | double |
| `arrivals_lag_24h` | double |
| `arrivals_lag_168h` | double |
| `net_flow_lag_1h` | double |
| `net_flow_lag_24h` | double |
| `net_flow_lag_168h` | double |
| `departures_rolling_mean_24h` | double |
| `departures_rolling_mean_168h` | double |

Las características históricas deben calcularse antes de invocar el modelo. La interfaz de inferencia no realiza automáticamente la generación de variables a partir de una estación y una fecha.

La salida corresponde a una estimación numérica continua de salidas por estación-hora.

## 4. Interfaz de inferencia

La implementación se encuentra en `mibici/modeling/predict.py` y proporciona dos funciones:

- `load_model(model_uri=DEFAULT_MODEL_URI)`: recupera el modelo desde MLflow.
- `predict(model, features)`: genera predicciones a partir de un DataFrame de pandas y devuelve una lista de valores `float`.

Ejemplo de integración:

```python
import pandas as pd

from mibici.modeling.predict import load_model, predict

model = load_model()

features = pd.DataFrame(
    [
        {
            "hour_of_day": 8,
            "day_of_week": 2,
            "month": 11,
            "is_weekend": False,
            "departures_lag_1h": 1.0,
            "departures_lag_24h": 2.0,
            "departures_lag_168h": 1.0,
            "arrivals_lag_1h": 1.0,
            "arrivals_lag_24h": 2.0,
            "arrivals_lag_168h": 1.0,
            "net_flow_lag_1h": 0.0,
            "net_flow_lag_24h": 0.0,
            "net_flow_lag_168h": 0.0,
            "departures_rolling_mean_24h": 1.5,
            "departures_rolling_mean_168h": 1.2,
        }
    ]
)

predictions = predict(model, features)
```

Los valores del ejemplo son ilustrativos y no representan una observación real del conjunto de datos.

## 5. Disponibilidad y recuperación del modelo

### 5.1. Persistencia y acceso

El modelo se administra mediante MLflow Model Registry, utilizando el nombre `mibici-p1-departures` y el alias `champion`.

La base de datos de seguimiento y los artefactos generados durante el entrenamiento están excluidos del control de versiones. En consecuencia, clonar el repositorio no proporciona automáticamente acceso al modelo registrado.

Para ejecutar inferencias en otro entorno, se requiere acceso a un servidor MLflow que disponga del registro y sus artefactos, o reproducir el entrenamiento y el registro localmente.

### 5.2. Reproducción del entrenamiento y registro

Desde la raíz del repositorio, con el entorno virtual y las dependencias instaladas, ejecutar:

```bash
mlflow run . -e p1_train --env-manager local --experiment-name mibici-equipo28
```

Este comando ejecuta los modelos de regresión P1 y registra sus métricas y artefactos en el experimento `mibici-equipo28`.

Posteriormente, registrar el mejor modelo mediante:

```bash
mlflow run . -e register --env-manager local --experiment-name mibici-equipo28
```

El procedimiento selecciona el modelo con menor `val_mae` y administra el alias `champion`. Cuando existe un modelo previamente seleccionado, el alias solo se actualiza si la nueva versión presenta una mejora estricta en la métrica de validación.

En la ejecución verificada se seleccionó `hist_gradient_boosting_poisson`, con un MAE de validación de `0.864087`. El registro automático creó la versión 2 y conservó la versión 1 como `champion`, debido a que ambas presentaron el mismo desempeño.

### 5.3. Integración con el servicio REST

La aplicación FastAPI debe recuperar el modelo utilizando:

```python
model_uri = "models:/mibici-p1-departures@champion"
```

Esta referencia permite utilizar el modelo seleccionado sin fijar una versión específica en el código.

El servicio debe disponer de acceso al Model Registry y a los artefactos correspondientes. Cuando se utilice un servidor MLflow, la variable `MLFLOW_TRACKING_URI` deberá configurarse con una dirección accesible desde el entorno de ejecución.

Para el despliegue mediante Docker o Podman, debe considerarse que `127.0.0.1` dentro de un contenedor hace referencia al propio contenedor y no necesariamente al equipo anfitrión.

La estrategia de distribución de los artefactos y conectividad con MLflow deberá definirse durante la implementación del contenedor, asegurando que el modelo pueda recuperarse durante la inicialización del servicio.

## 6. Validación realizada

La integración fue verificada mediante:

- Recuperación del modelo registrado con el alias `champion`.
- Confirmación de la firma de entrada de 15 características.
- Ejecución de cinco observaciones del ejemplo almacenado en MLflow.
- Comparación de las predicciones del módulo con las predicciones directas del modelo.
- Cinco pruebas unitarias exitosas mediante PyTest.
- Validación de calidad de código mediante Ruff.

Las pruebas unitarias utilizan objetos simulados para evitar depender de un servidor MLflow durante su ejecución.

## 7. Consideraciones para FastAPI

El servicio REST deberá convertir las características recibidas en un DataFrame de pandas compatible con la firma del modelo.

Se recomienda cargar el modelo una sola vez durante la inicialización del servicio y reutilizarlo para las solicitudes posteriores.

La validación de las solicitudes HTTP, los esquemas Pydantic y la serialización de respuestas corresponden a la implementación de FastAPI.

La integración debe conservar la precisión numérica de las predicciones y gestionar los errores derivados de entradas incompatibles o de la falta de acceso al Model Registry.