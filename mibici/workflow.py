"""Workflow de MLflow: encadena los pasos del proyecto bajo un run padre.

Por ahora solo incluye el paso `load`; los siguientes pasos se agregan
aquí conforme se implementen.

    mlflow run . -e workflow --env-manager local
"""

from loguru import logger
import mlflow
from mlflow.tracking import MlflowClient

ENV_MANAGER = "local"  # Reutilizar el ambiente actual en cada paso.


def main():
    """Ejecutar los pasos del proyecto como runs hijos de un run padre."""
    with mlflow.start_run() as parent:
        load = mlflow.run(".", "load", env_manager=ENV_MANAGER)

        # Pendiente: descomentar conforme se implementen los pasos. Cada paso
        # recibe el run_id del anterior para descargar sus artefactos.
        # features = mlflow.run(
        #     ".", "features", env_manager=ENV_MANAGER,
        #     parameters={"run_id": load.run_id},
        # )
        # train = mlflow.run(
        #     ".", "train", env_manager=ENV_MANAGER,
        #     parameters={"run_id": features.run_id},
        # )

        # Agrupar los pasos bajo el run padre en la UI de MLflow.
        # Agregar aquí cada paso nuevo: (load, features, train).
        client = MlflowClient()
        for step in (load,):
            client.set_tag(step.run_id, "mlflow.parentRunId", parent.info.run_id)

    logger.success(f"Workflow completado. Run de load: {load.run_id}")


if __name__ == "__main__":
    main()
