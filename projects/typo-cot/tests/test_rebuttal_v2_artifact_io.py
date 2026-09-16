"""Captured-byte and publication invariants for shared CPU artifact helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from typo_cot.experiments.rebuttal_v2 import artifact_io as io


def test_jsonl_preserves_unicode_line_separator_inside_value(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_bytes(io.jsonl_bytes([{"text": "a\u2028b\u0085c"}]))
    assert io.read_jsonl(path, {}) == [{"text": "a\u2028b\u0085c"}]


@pytest.mark.parametrize("raw", [b'{"x": 1, "x": 2}', b'{"x": NaN}', b'{"x": 1e9999}', b"[]"])
def test_json_reader_rejects_noncanonical_objects(tmp_path, raw):
    path = tmp_path / "input.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        io.read_json(path, {})


def test_changed_after_capture_aborts_publication_without_partial_output(tmp_path):
    source = tmp_path / "input.json"
    source.write_bytes(b'{"value": 1}')
    snapshots = {}
    assert io.read_json(source, snapshots) == {"value": 1}
    source.write_bytes(b'{"value": 2}')
    with pytest.raises(ValueError, match="changed after capture"):
        io.publish_artifacts(tmp_path / "output", {"result.json": b"{}"}, snapshots)
    assert not (tmp_path / "output").exists()
    assert list(tmp_path.glob(".output.audit-*")) == []


def test_checked_bytes_are_the_exact_bytes_returned_for_parsing(tmp_path, monkeypatch):
    path = tmp_path / "input.json"
    original = b'{"value": "captured"}'
    path.write_bytes(original)
    real_read = Path.read_bytes

    def changed_after_read(self):
        raw = real_read(self)
        if self == path:
            self.write_bytes(b'{"value": "later"}')
        return raw

    monkeypatch.setattr(Path, "read_bytes", changed_after_read)
    snapshots = {}
    assert io.read_json(path, snapshots) == {"value": "captured"}
    assert snapshots[path] == io.artifact_ref(path, original)["sha256"]
    with pytest.raises(ValueError, match="changed after capture"):
        io.revalidate_snapshots(snapshots)


@pytest.mark.parametrize("precreate_output", [False, True])
def test_private_atomic_output_never_overwrites_completed_audit(tmp_path, precreate_output):
    output = tmp_path / "output"
    if precreate_output:
        output.mkdir()
    io.publish_artifacts(output, {"run.json": b"{}"}, {})
    assert output.stat().st_mode & 0o777 == 0o700
    with pytest.raises(ValueError, match="absent or empty"):
        io.publish_artifacts(output, {"run.json": b"changed"}, {})
    assert (output / "run.json").read_bytes() == b"{}"


@pytest.mark.parametrize("precreate_output", [False, True])
def test_rename_failure_preserves_output_state_and_removes_owned_staging(
    tmp_path, monkeypatch, precreate_output
):
    output = tmp_path / "output"
    original_inode = None
    if precreate_output:
        output.mkdir()
        original_inode = output.stat().st_ino
    staging_paths = []

    def fail_rename(self, target):
        staging_paths.append(self)
        assert target == output
        raise OSError("injected rename failure")

    monkeypatch.setattr(Path, "rename", fail_rename)
    with pytest.raises(OSError, match="injected rename failure"):
        io.publish_artifacts(output, {"run.json": b"{}"}, {})

    assert output.exists() is precreate_output
    if precreate_output:
        assert output.is_dir()
        assert output.stat().st_ino == original_inode
        assert list(output.iterdir()) == []
    assert len(staging_paths) == 1
    assert not staging_paths[0].exists()
    assert list(tmp_path.glob(".output.audit-*")) == []


def test_concurrent_nonempty_output_is_preserved(tmp_path, monkeypatch):
    output = tmp_path / "output"

    def create_concurrent_output(_snapshots):
        output.mkdir()
        (output / "concurrent.txt").write_text("keep\n", encoding="utf-8")

    monkeypatch.setattr(io, "revalidate_snapshots", create_concurrent_output)
    with pytest.raises(OSError):
        io.publish_artifacts(output, {"run.json": b"{}"}, {})

    assert (output / "concurrent.txt").read_text(encoding="utf-8") == "keep\n"
    assert list(tmp_path.glob(".output.audit-*")) == []


@pytest.mark.parametrize("name", ["../escape.json", "/absolute.json", "nested/file.json", "..", ""])
def test_published_artifact_names_cannot_escape_staging(tmp_path, name):
    with pytest.raises(ValueError, match="flat file names"):
        io.publish_artifacts(tmp_path / "out", {name: b"{}"}, {})
    assert not (tmp_path / "out").exists()


def test_csv_view_escapes_formulas_without_mutating_json_value():
    rows = [{"value": " \t=HYPERLINK(1)", "missing": None}]
    csv = io.csv_bytes(rows, ["value", "missing"])
    assert b"' \t=HYPERLINK(1)" in csv
    assert b"NA" in csv
    assert json.loads(io.json_bytes(rows))[0]["value"] == " \t=HYPERLINK(1)"
