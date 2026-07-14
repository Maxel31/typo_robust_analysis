"""パッケージの基本テスト."""


def test_package_import() -> None:
    """パッケージが正常にインポートできることを確認."""
    import typo_cot

    assert typo_cot.__version__ == "0.1.0"


def test_subpackages_import() -> None:
    """サブパッケージが正常にインポートできることを確認."""
    from typo_cot import data, evaluation, lrp, models, perturbation, visualization

    # サブパッケージが存在することを確認
    assert data is not None
    assert models is not None
    assert lrp is not None
    assert perturbation is not None
    assert evaluation is not None
    assert visualization is not None
