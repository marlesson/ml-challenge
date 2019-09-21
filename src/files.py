import os
import json
from typing import Type


def get_params_path(task_dir: str) -> str:
    return os.path.join(task_dir, "params.json")


def get_params(task_dir: str) -> dict:
    with open(get_params_path(task_dir), "r") as params_file:
        return json.load(params_file)


def get_classes_path(task_dir: str) -> str:
    return os.path.join(task_dir, "classes.json")


def get_torch_weights_path(task_dir: str) -> str:
    return os.path.join(task_dir, "weights.pt")


def get_keras_weights_path(task_dir: str) -> str:
    return os.path.join(task_dir, "weights.h5")


def get_history_path(task_dir: str) -> str:
    return os.path.join(task_dir, "history.csv")

def get_submission_path(task_dir: str) -> str:
    return os.path.join(task_dir, "submission.csv")

def get_history_plot_path(task_dir: str) -> str:
    return os.path.join(task_dir, "history.jpg")


def get_tensorboard_logdir(task_id: str) -> str:
    return os.path.join("output", "tensorboard_logs", task_id)


def get_task_dir(model_cls: Type, task_id: str):
    return os.path.join("output", "models", model_cls.__name__, "results", task_id)


def get_operadoras_path():
    return os.path.join("input", "operadoras.csv")