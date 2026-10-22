from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


@dataclass(frozen=True)
class Settings:
    input_folder: Path = Path(r"C:\Users\USER\Pictures\Screenshots")
    model_name: str = "gemma3:4b"
    blur_threshold: float = 100.0
    dark_threshold: float = 40.0
    bright_threshold: float = 215.0
    duplicate_hash_distance: int = 5
    image_size: int = 768


def load_settings(config_path: Path | None = None) -> Settings:
    config_file = config_path or Path("config.toml")
    if not config_file.exists():
        return Settings()

    data = _read_toml(config_file)
    return Settings(
        input_folder=Path(data.get("input_folder", Settings.input_folder)),
        model_name=str(data.get("model_name", Settings.model_name)),
        blur_threshold=float(data.get("blur_threshold", Settings.blur_threshold)),
        dark_threshold=float(data.get("dark_threshold", Settings.dark_threshold)),
        bright_threshold=float(data.get("bright_threshold", Settings.bright_threshold)),
        duplicate_hash_distance=int(
            data.get("duplicate_hash_distance", Settings.duplicate_hash_distance)
        ),
        image_size=int(data.get("image_size", Settings.image_size)),
    )


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as fp:
        data = tomllib.load(fp)
    if not isinstance(data, dict):
        return {}
    return data
