import hashlib
import json

import pytest
import yaml

from mibici.api.preflight import verify_export, verify_reference


@pytest.fixture
def export(tmp_path):
    entries = []
    expected = {}
    for key in ("bike", "dock"):
        folder = f"p2-{key}-pressure"
        name = f"mibici-{folder}"
        run_id = key * 8
        expected[name] = ("1", run_id)
        entries.append(
            {
                "registered_model": name,
                "alias": "champion",
                "version": "1",
                "run_id": run_id,
                "path": folder,
            }
        )
        model = tmp_path / folder
        model.mkdir()
        for filename in ("model.skops", "requirements.txt", "python_env.yaml", "conda.yaml"):
            (model / filename).write_text("fixture", encoding="utf-8")
        (model / "MLmodel").write_text(
            yaml.safe_dump(
                {
                    "run_id": run_id,
                    "flavors": {
                        "sklearn": {
                            "serialization_format": "skops",
                            "pickled_model": "model.skops",
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
    (tmp_path / "metadata.json").write_text(
        json.dumps({"models": entries, "tracking_uri": "not-for-evidence"}), encoding="utf-8"
    )
    return tmp_path, expected


def test_preflight_hashes_export_without_exposing_tracking_uri(export):
    root, expected = export
    result = verify_export(root, "champion", expected)
    assert {row["registered_model"] for row in result["models"]} == set(expected)
    assert (
        result["sha256"]["p2-bike-pressure/model.skops"] == hashlib.sha256(b"fixture").hexdigest()
    )
    assert "not-for-evidence" not in json.dumps(result)


def test_rejects_same_version_from_different_run(export):
    root, expected = export
    expected["mibici-p2-bike-pressure"] = ("1", "different")
    with pytest.raises(ValueError, match="mibici-p2-bike-pressure.*run_id"):
        verify_export(root, "champion", expected)


def test_rejects_nested_artifact(export):
    root, expected = export
    bike = root / "p2-bike-pressure"
    (bike / "artifacts").mkdir()
    (bike / "model.skops").rename(bike / "artifacts" / "model.skops")
    with pytest.raises(ValueError, match="p2-bike-pressure/model.skops"):
        verify_export(root, "champion", expected)


def test_rejects_mlmodel_from_different_run(export):
    root, expected = export
    path = root / "p2-dock-pressure" / "MLmodel"
    descriptor = yaml.safe_load(path.read_text(encoding="utf-8"))
    descriptor["run_id"] = "different"
    path.write_text(yaml.safe_dump(descriptor), encoding="utf-8")
    with pytest.raises(ValueError, match="p2-dock-pressure/MLmodel: run_id"):
        verify_export(root, "champion", expected)


def test_rejects_missing_metadata(export):
    root, expected = export
    (root / "metadata.json").unlink()
    with pytest.raises(ValueError, match="metadata.json"):
        verify_export(root, "champion", expected)


@pytest.mark.parametrize("change", ["modified", "missing", "extra"])
def test_rejects_different_contents_with_same_lineage(export, change):
    root, expected = export
    auxiliary = root / "p2-bike-pressure" / "input_example.json"
    if change != "extra":
        auxiliary.write_text("original", encoding="utf-8")
    reference = verify_export(root, "champion", expected)
    verify_reference(reference, reference)

    if change == "missing":
        auxiliary.unlink()
    else:
        auxiliary.write_text("different", encoding="utf-8")
    with pytest.raises(ValueError, match="p2-bike-pressure/input_example.json"):
        verify_reference(verify_export(root, "champion", expected), reference)
