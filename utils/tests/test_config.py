"""Regression tests for typo_utils.config.load_config `_base_` 継承."""
from typo_utils.config import load_config


def test_base_merge_and_override(tmp_path):
    (tmp_path / "base.yaml").write_text("a: 1\nb: 2\nnested:\n  x: 10\n  y: 20\n")
    (tmp_path / "child.yaml").write_text(
        "_base_: base.yaml\nb: 99\nnested:\n  y: 200\nc: 3\n"
    )
    cfg = load_config(tmp_path / "child.yaml")
    assert cfg.a == 1            # base から継承
    assert cfg.b == 99           # child が base を上書き
    assert cfg.nested.x == 10    # nested も継承
    assert cfg.nested.y == 200   # nested も child が上書き
    assert cfg.c == 3            # child のみ
    assert "_base_" not in cfg   # _base_ キーは除去

    cfg2 = load_config(tmp_path / "child.yaml", ["b=7", "nested.x=11"])
    assert cfg2.b == 7           # CLI override が最優先
    assert cfg2.nested.x == 11


def test_plain_load_without_base(tmp_path):
    (tmp_path / "plain.yaml").write_text("k: v\n")
    cfg = load_config(tmp_path / "plain.yaml")
    assert cfg.k == "v"
    assert "_base_" not in cfg


def test_nested_base_chain(tmp_path):
    (tmp_path / "g.yaml").write_text("root: 1\n")
    (tmp_path / "mid.yaml").write_text("_base_: g.yaml\nmid: 2\n")
    (tmp_path / "leaf.yaml").write_text("_base_: mid.yaml\nleaf: 3\n")
    cfg = load_config(tmp_path / "leaf.yaml")
    assert cfg.root == 1
    assert cfg.mid == 2
    assert cfg.leaf == 3


def test_list_base(tmp_path):
    (tmp_path / "a.yaml").write_text("a: 1\nshared: from_a\n")
    (tmp_path / "b.yaml").write_text("b: 2\nshared: from_b\n")
    (tmp_path / "c.yaml").write_text("_base_: [a.yaml, b.yaml]\nc: 3\n")
    cfg = load_config(tmp_path / "c.yaml")
    assert cfg.a == 1 and cfg.b == 2 and cfg.c == 3
    assert cfg.shared == "from_b"   # 後勝ち（リスト後方が上書き）
