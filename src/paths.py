# src/paths.py
import os


def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_data_dir():
    return os.path.join(get_project_root(), "data")


def get_models_dir():
    models_dir = os.path.join(get_project_root(), "models")
    os.makedirs(models_dir, exist_ok=True)
    return models_dir


def get_results_dir():
    results_dir = os.path.join(get_project_root(), "results")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir
