from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from openai import OpenAI

_ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _default_models_path() -> Path:
    return Path(__file__).resolve().parents[4] / "models.yaml"


def _expand_env_value(value: Any) -> Any:
    if isinstance(value, str):
        def _replace(match: re.Match) -> str:
            env_key = match.group(1)
            env_val = os.getenv(env_key)
            if env_val is None:
                raise ValueError(f"Environment variable '{env_key}' is not set")
            return env_val

        return _ENV_PATTERN.sub(_replace, value)
    if isinstance(value, dict):
        return {k: _expand_env_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_value(v) for v in value]
    return value


@lru_cache(maxsize=4)
def load_models_registry(models_path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    path = Path(models_path) if models_path else _default_models_path()
    if not path.exists():
        raise FileNotFoundError(f"models.yaml not found at: {path}")

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    models = data.get("models", [])
    if not isinstance(models, list):
        raise ValueError("models.yaml must contain a top-level 'models' list")

    registry: Dict[str, Dict[str, Any]] = {}
    for entry in models:
        if not isinstance(entry, dict) or "name" not in entry:
            continue
        expanded = _expand_env_value(entry)
        registry[expanded["name"]] = expanded

    return registry


def get_model_config(model_name: str, models_path: Optional[str] = None) -> Dict[str, Any]:
    registry = load_models_registry(models_path)
    if model_name not in registry:
        available = ", ".join(sorted(registry.keys()))
        raise ValueError(
            f"Model '{model_name}' not found in models.yaml. Available models: {available}"
        )
    return registry[model_name]


def get_model_request_params(model_config: Dict[str, Any]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for key in ("temperature", "max_tokens", "extra_body"):
        if key in model_config and model_config[key] is not None:
            params[key] = model_config[key]
    return params


def create_openai_client(model_config: Dict[str, Any], timeout: Optional[float] = None) -> OpenAI:
    api_key = model_config.get("api_key")
    base_url = model_config.get("base_url")
    if not api_key:
        raise ValueError("api_key is required in models.yaml for the selected model")
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    if timeout is not None:
        client_kwargs["timeout"] = timeout
    return OpenAI(**client_kwargs)
