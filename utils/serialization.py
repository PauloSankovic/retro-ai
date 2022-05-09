import logging
import os
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

DIR = Path(__file__).parent.parent


def create_path(base_dir: str, model_name: str, parts: dict) -> Path:
    name_parts = [f"{key}={value}" for key, value in parts.items()]
    name_parts.insert(0, model_name)
    return DIR / base_dir / "_".join(name_parts)


def save_model(model, base_dir: str = "models", **kwargs) -> None:
    path = create_path(base_dir, model.name, kwargs)
    with open(f"{path}.pickle", "wb") as f:
        try:
            logger.info(f"Saving model at location {os.path.realpath(f.name)}")
            pickle.dump(model, f)
        except Exception as e:
            logger.error(f"Error while saving model {model.name}: {e}")


def load_model(model_name: str, base_dir: str = "models", **kwargs) -> object:
    path = create_path(base_dir, model_name, kwargs)
    with open(f"{path}.pickle", "rb") as f:
        return pickle.load(f)
