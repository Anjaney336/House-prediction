from __future__ import annotations

import importlib
import inspect
from types import ModuleType


def require_module_api(
    module_name: str,
    required_parameters: dict[str, set[str]] | None = None,
    required_attributes: set[str] | None = None,
) -> ModuleType:
    """Reload a stale Streamlit-cached module when its public API is outdated.

    Streamlit reruns page scripts but can retain imported workspace modules in
    ``sys.modules``. During local upgrades that can leave a new page bound to
    an older function object. The guard is capability-based and reloads only
    when the loaded module does not satisfy the API required by the page.
    """
    module = importlib.import_module(module_name)

    def is_compatible(candidate: ModuleType) -> bool:
        for attribute in required_attributes or set():
            if not hasattr(candidate, attribute):
                return False
        for function_name, parameters in (required_parameters or {}).items():
            function = getattr(candidate, function_name, None)
            if function is None or not parameters.issubset(inspect.signature(function).parameters):
                return False
        return True

    if not is_compatible(module):
        module = importlib.reload(module)
    if not is_compatible(module):
        raise RuntimeError(f"Workspace module '{module_name}' does not provide the required application API.")
    return module
