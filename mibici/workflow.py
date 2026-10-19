"""Workflow de MLflow: encadena los pasos del proyecto bajo un run padre.

Ejecuta la ingeniería de datos, los modelos P1 y la preparación y los modelos
P2. Los pasos de modelado leen solo train/validation (enero-octubre).

    mlflow run . -e workflow --env-manager local
"""

from loguru import logger
import mlflow
from mlflow.tracking import MlflowClient

ENV_MANAGER = "local"  # Reutilizar el ambiente actual en cada paso.

# Orden de ejecución: cada paso consume los archivos que genera el anterior.
STEPS = ("load", "p1_train", "p2_catalog", "p2_targets", "p2_train")


def main():
    """Ejecutar los pasos del proyecto como runs hijos de un run padre."""
    with mlflow.start_run() as parent:
        steps = [mlflow.run(".", step, env_manager=ENV_MANAGER) for step in STEPS]

        # Agrupar los pasos bajo el run padre en la UI de MLflow.
        client = MlflowClient()
        for step in steps:
            client.set_tag(step.run_id, "mlflow.parentRunId", parent.info.run_id)

    logger.success(f"Workflow completado: {len(steps)} pasos.")


if __name__ == "__main__":
    main()
