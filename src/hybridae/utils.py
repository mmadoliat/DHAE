"""Small shared utilities."""
from __future__ import annotations
import json, os, random, platform
from pathlib import Path
import numpy as np
import torch


def set_seed(seed: int, single_thread: bool = False) -> None:
    """Seed python, numpy and torch (as in the original notebooks)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if single_thread:
        torch.set_num_threads(1)


def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def environment_info() -> dict:
    import scipy, sklearn, statsmodels, pandas
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
        "sklearn": sklearn.__version__, "statsmodels": statsmodels.__version__,
        "pandas": pandas.__version__,
    }


def save_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=conv)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]
