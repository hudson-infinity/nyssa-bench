from __future__ import annotations

import hashlib
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType


def load_module_from_path(path: str | Path) -> ModuleType:
    """Execute a file with a registered, path-specific module identity."""
    path = Path(path).resolve()
    digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()
    name = f"_nyssa_user_module_{digest}"
    spec = spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load module from {path}")
    module = module_from_spec(spec)
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
        raise
    return module
