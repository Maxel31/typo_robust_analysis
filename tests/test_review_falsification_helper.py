from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
HELPER = REPOSITORY_ROOT / ".github" / "review" / "run-falsification-test.sh"


@pytest.fixture(autouse=True)
def _restore_temporary_permissions(tmp_path: Path):
    """Undo the fake root-ownership mode before pytest removes tmp_path."""

    yield
    paths = sorted(tmp_path.rglob("*"), key=lambda path: len(path.parts), reverse=True)
    for path in paths:
        if not path.is_symlink():
            path.chmod(path.stat().st_mode | stat.S_IWUSR)
    tmp_path.chmod(tmp_path.stat().st_mode | stat.S_IWUSR)


def _write_executable(path: Path, contents: str) -> None:
    path.write_text(contents, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _make_helper(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, str]]:
    state = tmp_path / "state"
    review_tests = tmp_path / "review-tests"
    helper = tmp_path / "review-falsify"
    source = HELPER.read_text(encoding="utf-8")
    source = source.replace(
        'readonly REVIEW_STATE_DIR="/var/lib/claude-pr-review"',
        f'readonly REVIEW_STATE_DIR="{state}"',
    ).replace(
        'readonly REVIEW_TEST_ROOT="/tmp/claude-review-tests"',
        f'readonly REVIEW_TEST_ROOT="{review_tests}"',
    )
    _write_executable(helper, source)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "sudo",
        """#!/usr/bin/env bash
set -euo pipefail
command_name="$1"
shift
case "${command_name}" in
  install)
    args=()
    while (($#)); do
      case "$1" in
        -o|-g) shift 2 ;;
        *) args+=("$1"); shift ;;
      esac
    done
    command install "${args[@]}"
    ;;
  chown)
    target="${@: -1}"
    chmod -R a-w -- "${target}"
    ;;
  rm)
    for target in "$@"; do
      if [[ "${target}" != -* && "${target}" != -- && -e "${target}" ]]; then
        chmod -R u+w -- "${target}"
      fi
    done
    command rm "$@"
    ;;
  tee)
    target="${@: -1}"
    [[ ! -e "${target}" ]] || chmod u+w -- "${target}"
    command tee "$@"
    ;;
  *) command "${command_name}" "$@" ;;
esac
""",
    )
    _write_executable(
        fake_bin / "docker",
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >>"${FAKE_DOCKER_LOG}"
if [[ "${1:-}" == pull ]]; then
  exit 0
fi
if [[ " $* " == *" --entrypoint /bin/bash "* ]]; then
  mkdir -p "${FAKE_REVIEW_ENV_ROOT}/shared/bin"
  printf 'home = /container-only\n' >"${FAKE_REVIEW_ENV_ROOT}/shared/pyvenv.cfg"
  ln -s /container-only/python3.12 "${FAKE_REVIEW_ENV_ROOT}/shared/bin/python"
fi
exit 0
""",
    )
    environment = os.environ.copy()
    environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
    environment["FAKE_REVIEW_ENV_ROOT"] = str(state / "envs")
    environment["FAKE_DOCKER_LOG"] = str(tmp_path / "docker.log")
    return helper, state, review_tests, environment


def _run(
    helper: Path, environment: dict[str, str], *arguments: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(helper), *arguments],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )


def _prepare(helper: Path, environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    base_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD^"], cwd=REPOSITORY_ROOT, text=True
    ).strip()
    return _run(helper, environment, "--prepare", str(REPOSITORY_ROOT), base_sha)


def test_prepare_is_rerunnable(tmp_path: Path) -> None:
    helper, _, _, environment = _make_helper(tmp_path)

    first = _prepare(helper, environment)
    second = _prepare(helper, environment)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr


def test_run_accepts_container_relative_venv_symlink(tmp_path: Path) -> None:
    helper, _, review_tests, environment = _make_helper(tmp_path)
    prepared = _prepare(helper, environment)
    assert prepared.returncode == 0, prepared.stderr
    probe = review_tests / "test_probe.py"
    probe.write_text("def test_ok():\n    assert True\n", encoding="utf-8")

    result = _run(helper, environment, "root", str(probe))

    assert result.returncode == 0, result.stderr
    docker_log = Path(environment["FAKE_DOCKER_LOG"]).read_text(encoding="utf-8")
    assert "dst=/workspace/tests/.review-tests,readonly" in docker_log
    assert "/workspace/tests/.review-tests/test_probe.py" in docker_log


def test_review_test_mount_points_exist_and_are_real_directories() -> None:
    expected = (
        REPOSITORY_ROOT / "tests" / ".review-tests",
        REPOSITORY_ROOT / "projects" / "typo-cot" / "tests" / ".review-tests",
    )

    for mount_point in expected:
        assert mount_point.is_dir()
        assert not mount_point.is_symlink()


def test_run_rejects_a_symlinked_review_test_mount(tmp_path: Path) -> None:
    helper, _, review_tests, environment = _make_helper(tmp_path)
    prepared = _prepare(helper, environment)
    assert prepared.returncode == 0, prepared.stderr
    probe = review_tests / "test_probe.py"
    probe.write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    (workspace / "tests").mkdir(parents=True)
    (workspace / "redirect").mkdir()
    (workspace / "tests" / ".review-tests").symlink_to(workspace / "redirect")
    (workspace / ".git").mkdir()
    state_workspace = tmp_path / "state" / "workspace"
    state_workspace.chmod(state_workspace.stat().st_mode | stat.S_IWUSR)
    state_workspace.write_text(f"{workspace}\n", encoding="utf-8")

    result = _run(helper, environment, "root", str(probe))

    assert result.returncode == 2
    assert "not a real directory" in result.stderr


def test_self_test_fails_cleanly_before_prepare(tmp_path: Path) -> None:
    helper, _, _, environment = _make_helper(tmp_path)

    result = _run(helper, environment, "--self-test")

    assert result.returncode == 2
    assert "sandbox is not prepared" in result.stderr


def test_self_test_arms_sentinel_and_always_removes_probe(tmp_path: Path) -> None:
    helper, _, review_tests, environment = _make_helper(tmp_path)
    prepared = _prepare(helper, environment)
    assert prepared.returncode == 0, prepared.stderr

    result = _run(helper, environment, "--self-test")

    assert result.returncode == 0, result.stderr
    assert "export REVIEW_SANDBOX_SENTINEL=" in helper.read_text(encoding="utf-8")
    assert not (review_tests / "test_review_sandbox.py").exists()
    assert stat.S_IMODE(review_tests.stat().st_mode) == 0o755


def test_dependency_build_storage_is_disk_backed() -> None:
    source = HELPER.read_text(encoding="utf-8")

    assert '--mount "type=bind,src=${REVIEW_BUILD_ROOT},dst=/review-build"' in source
    assert "--env TMPDIR=/review-build/tmp" in source
    assert "--env UV_CACHE_DIR=/review-build/cache" in source


def test_sandbox_checks_network_interfaces_and_rejects_timeouts() -> None:
    source = HELPER.read_text(encoding="utf-8")

    assert "socket.if_nameindex()" in source
    assert 'interfaces <= {"lo"}' in source
    assert "except TimeoutError as error:" in source
    assert "error.errno in allowed" in source


def test_prepare_and_test_execution_have_wall_clock_limits() -> None:
    source = HELPER.read_text(encoding="utf-8")

    assert '"${REVIEW_PREPARE_TIMEOUT_SECONDS}s"' in source
    assert '"${REVIEW_TEST_TIMEOUT_SECONDS}s"' in source


def test_missing_test_path_is_a_usage_error(tmp_path: Path) -> None:
    helper, _, review_tests, environment = _make_helper(tmp_path)
    prepared = _prepare(helper, environment)
    assert prepared.returncode == 0, prepared.stderr

    result = _run(helper, environment, "root", str(review_tests / "missing.py"))

    assert result.returncode == 2
    assert result.stderr.startswith("review-falsify:")
