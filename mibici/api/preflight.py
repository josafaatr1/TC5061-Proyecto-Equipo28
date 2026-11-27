"""Comprueba un export P2 antes del build, sin cargar modelos ni consultar MLflow."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path
from typing import TypedDict

import yaml

from mibici.api.export_models import DEFAULT_ALIAS, EXPORTED_MODELS
from mibici.config import MODELS_DIR


class ExportReport(TypedDict):
    models: list[dict[str, str]]
    sha256: dict[str, str]


def require_file(root: Path, relative: str) -> Path:
    path = root / relative
    if not path.is_file() or not path.stat().st_size or path.resolve() != root / relative:
        raise ValueError(f"Archivo ausente, vacío o redirigido: {relative}")
    return path


def verify_export(root: Path, alias: str, expected: dict[str, tuple[str, str]]) -> ExportReport:
    root = root.resolve()
    metadata = require_file(root, "metadata.json")
    try:
        manifest = json.loads(metadata.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise ValueError("metadata.json inválido") from exc

    entries = manifest.get("models") if isinstance(manifest, dict) else None
    if not isinstance(entries, list) or len(entries) != len(EXPORTED_MODELS):
        raise ValueError("metadata.json debe describir exactamente los dos modelos P2")

    files = {"metadata.json": metadata}
    lineage = []
    for name, folder in EXPORTED_MODELS.items():
        version, run_id = expected[name]
        matches = [e for e in entries if isinstance(e, dict) and e.get("registered_model") == name]
        wanted = {
            "registered_model": name,
            "alias": alias,
            "version": version,
            "run_id": run_id,
            "path": folder,
        }
        if len(matches) != 1 or any(matches[0].get(k) != v for k, v in wanted.items()):
            raise ValueError(f"{name}: alias, versión, run_id o ruta no coincide con lo esperado")

        descriptor = f"{folder}/MLmodel"
        files[descriptor] = require_file(root, descriptor)
        try:
            model = yaml.safe_load(files[descriptor].read_text(encoding="utf-8"))
        except (yaml.YAMLError, UnicodeError) as exc:
            raise ValueError(f"{descriptor}: YAML inválido") from exc
        flavors = model.get("flavors") if isinstance(model, dict) else None
        sklearn = flavors.get("sklearn") if isinstance(flavors, dict) else None
        if (
            not isinstance(sklearn, dict)
            or model.get("run_id") != run_id
            or sklearn.get("serialization_format") != "skops"
            or sklearn.get("pickled_model") != "model.skops"
        ):
            raise ValueError(f"{descriptor}: run_id o formato sklearn/skops no coincide")
        for filename in ("model.skops", "requirements.txt", "python_env.yaml", "conda.yaml"):
            relative = f"{folder}/{filename}"
            require_file(root, relative)
        for path in (root / folder).rglob("*"):
            relative = path.relative_to(root).as_posix()
            if path.is_symlink() or path.is_junction():
                raise ValueError(f"El export no admite enlaces o junctions: {relative}")
            if not path.is_dir():
                files[relative] = require_file(root, relative)
        lineage.append(wanted)

    hashes = {}
    for relative, path in files.items():
        with path.open("rb") as stream:
            hashes[relative] = file_digest(stream, "sha256").hexdigest()
    return {"models": lineage, "sha256": hashes}


def verify_reference(report: ExportReport, reference: object) -> None:
    if not isinstance(reference, dict) or reference.get("models") != report["models"]:
        raise ValueError("El linaje no coincide con el informe SHA256 de referencia")
    expected = reference.get("sha256")
    if not isinstance(expected, dict):
        raise TypeError("El informe de referencia no contiene un mapa sha256")
    actual = report["sha256"]
    changed = sorted(
        path for path in expected.keys() | actual.keys() if expected.get(path) != actual.get(path)
    )
    if changed:
        raise ValueError(
            f"Archivos distintos respecto a la referencia SHA256: {', '.join(changed)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--alias", default=DEFAULT_ALIAS)
    parser.add_argument("--bike-version", required=True)
    parser.add_argument("--bike-run-id", required=True)
    parser.add_argument("--dock-version", required=True)
    parser.add_argument("--dock-run-id", required=True)
    parser.add_argument(
        "--expected-report", type=Path, help="Informe SHA256 verificado previamente"
    )
    args = parser.parse_args()
    expected = {
        "mibici-p2-bike-pressure": (args.bike_version, args.bike_run_id),
        "mibici-p2-dock-pressure": (args.dock_version, args.dock_run_id),
    }
    try:
        result = verify_export(args.models_dir, args.alias, expected)
        if args.expected_report is not None:
            verify_reference(result, json.loads(args.expected_report.read_text(encoding="utf-8")))
    except (OSError, TypeError, ValueError) as exc:
        parser.exit(1, f"Preflight falló: {exc}\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
