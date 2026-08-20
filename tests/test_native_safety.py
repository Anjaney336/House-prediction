from types import SimpleNamespace

from src.models import native_safety


def test_native_dependency_crash_isolated_from_main_process(monkeypatch):
    native_safety.probe_dependency.cache_clear()
    monkeypatch.setattr(native_safety.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=-1073741819, stderr="access violation", stdout=""))
    result = native_safety.probe_dependency("lightgbm")
    assert not result.safe
    assert "isolated preflight" in result.detail
    native_safety.probe_dependency.cache_clear()
