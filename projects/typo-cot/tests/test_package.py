"""パッケージの基本テスト."""

import subprocess
import sys


def test_package_import() -> None:
    """パッケージが正常にインポートできることを確認."""
    import typo_cot

    assert typo_cot.__version__ == "0.1.0"


def test_subpackages_import() -> None:
    """サブパッケージが正常にインポートできることを確認."""
    from typo_cot import data, evaluation, experiments, lrp, models

    # サブパッケージが存在することを確認
    assert data is not None
    assert models is not None
    assert lrp is not None
    assert evaluation is not None
    assert experiments is not None


def test_base_cli_import_does_not_require_the_lrp_numpy_extra() -> None:
    script = r"""
import builtins

original_import = builtins.__import__

def without_numpy(name, *args, **kwargs):
    if name == "numpy" or name.startswith("numpy."):
        raise ModuleNotFoundError("numpy is intentionally unavailable")
    return original_import(name, *args, **kwargs)

builtins.__import__ = without_numpy
import typo_cot.cli  # noqa: F401
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
