# src/run_metadata.py
"""Sidecar metadata for prediction outputs (JSON next to result CSV)."""
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

DECISION_RULE = {
    "description": (
        "A label is assigned if its probability >= threshold. "
        "If no label reaches the threshold, keep the single top-1 (argmax) label "
        "so every molecule receives at least one label."
    ),
    "threshold": 0.5,
    "comparator": ">=",
    "fallback": "argmax / top-1 when all probabilities < threshold",
    "decision_function": "src.data_utils.probs_to_multilabel_preds",
}

MODEL_ARCHITECTURE = {
    "name": "GINGGNNModel",
    "node_features_dim": 1,
    "hidden_dim": 64,
    "fingerprint_dim": 4263,
    "dropout_rate": 0.5,
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _package_version(dist_name):
    try:
        from importlib import metadata
        return metadata.version(dist_name)
    except Exception:
        return None


def collect_software_versions():
    mapping = {
        "numpy": "numpy",
        "pandas": "pandas",
        "scikit-learn": "scikit-learn",
        "scipy": "scipy",
        "rdkit": "rdkit",
        "torch": "torch",
        "torch-geometric": "torch-geometric",
    }
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for key, dist in mapping.items():
        versions[key] = _package_version(dist)
    # rdkit may be installed as rdkit-pypi
    if versions.get("rdkit") is None:
        versions["rdkit"] = _package_version("rdkit-pypi")
    return versions


def resolve_code_commit(project_root):
    git_dir = os.path.join(project_root, ".git")
    if os.path.isdir(git_dir):
        try:
            sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=project_root,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            return {"git_sha": sha, "git_sha_short": sha[:7], "source": "local_git"}
        except Exception:
            pass

    meta_path = os.path.join(project_root, "reproducibility_metadata.json")
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            commit = meta.get("project", {}).get("code_commit", {})
            if commit.get("git_sha"):
                return {
                    "git_sha": commit.get("git_sha"),
                    "git_sha_short": commit.get("git_sha_short"),
                    "ref": commit.get("ref"),
                    "source": "reproducibility_metadata.json",
                }
        except Exception:
            pass
    return None


def build_prediction_run_metadata(
    *,
    project_root,
    entry_script,
    input_csv,
    output_csv,
    n_molecules_valid,
    device,
    levels,
    model_config,
    batch_size=64,
    threshold=0.5,
):
    models = {}
    for level in levels:
        cfg = model_config[level]
        model_path = cfg["model_path"]
        mlb_path = cfg["mlb_path"]
        models[level] = {
            "model_path": os.path.relpath(model_path, project_root).replace("\\", "/"),
            "model_sha256": sha256_file(model_path) if os.path.isfile(model_path) else None,
            "mlb_path": os.path.relpath(mlb_path, project_root).replace("\\", "/"),
            "mlb_sha256": sha256_file(mlb_path) if os.path.isfile(mlb_path) else None,
        }

    decision = dict(DECISION_RULE)
    decision["threshold"] = threshold

    meta = {
        "schema_id": "DeepNPClass.prediction_run_metadata.v1",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "timezone_policy": {
            "all_timestamps": "UTC",
            "format": "ISO-8601 with Z suffix (UTC)",
        },
        "purpose": (
            "Sidecar metadata for a DeepNPClass prediction output: decision rules, "
            "execution time, software/model versions, and configuration parameters."
        ),
        "entry_script": entry_script,
        "executed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "device": str(device),
        "input": {
            "path": os.path.relpath(input_csv, project_root).replace("\\", "/"),
            "sha256": sha256_file(input_csv) if os.path.isfile(input_csv) else None,
            "n_molecules_valid": int(n_molecules_valid),
        },
        "output": {
            "path": os.path.relpath(output_csv, project_root).replace("\\", "/"),
            "sha256": sha256_file(output_csv) if os.path.isfile(output_csv) else None,
        },
        "decision_rule": decision,
        "model_architecture": MODEL_ARCHITECTURE,
        "configuration": {
            "levels": list(levels),
            "batch_size": batch_size,
            "threshold": threshold,
        },
        "models": models,
        "software_versions": collect_software_versions(),
        "code_commit": resolve_code_commit(project_root),
        "command": " ".join(sys.argv),
    }
    return meta


def write_prediction_meta(meta, meta_path):
    os.makedirs(os.path.dirname(os.path.abspath(meta_path)) or ".", exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return meta_path


def meta_path_for_output_csv(output_csv):
    base, _ = os.path.splitext(output_csv)
    return base + ".meta.json"
