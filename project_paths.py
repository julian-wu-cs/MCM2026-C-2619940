from pathlib import Path
from typing import Optional


def get_project_root() -> Path:
    return Path(__file__).resolve().parent


def get_data_path(data_arg: Optional[str]) -> Path:
    if data_arg:
        return Path(data_arg)
    return get_project_root() / "Data" / "2026_MCM_Problem_C_Data.csv"


def get_out_dir(q: str, out_arg: Optional[str]) -> Path:
    if out_arg:
        return Path(out_arg)
    return get_project_root() / "outputs" / q


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
