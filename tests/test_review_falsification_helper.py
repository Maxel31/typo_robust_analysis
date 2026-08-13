from __future__ import annotations

import os
import shutil
import stat
import subprocess
import time
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
HELPER = REPOSITORY_ROOT / ".github" / "review" / "run-falsification-test.sh"
REVIEW_IMAGE = "ghcr.io/astral-sh/uv:python3.12-bookworm"


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


def _docker_image_is_ready() -> bool:
    if shutil.which("docker") is None:
        return False
    daemon = subprocess.run(
        ["docker", "info"], check=False, capture_output=True, text=True
    )
    image = subprocess.run(
        ["docker", "image", "inspect", REVIEW_IMAGE],
        check=False,
        capture_output=True,
        text=True,
    )
    return daemon.returncode == 0 and image.returncode == 0


def _make_real_docker_helper(
    tmp_path: Path, *, timeout_seconds: int = 600, kill_after_seconds: int = 5
) -> tuple[Path, Path, dict[str, str]]:
    state = tmp_path / "real-state"
    review_tests = tmp_path / "real-review-tests"
    helper = tmp_path / "real-review-falsify"
    source = HELPER.read_text(encoding="utf-8")
    source = source.replace(
        'readonly REVIEW_STATE_DIR="/var/lib/claude-pr-review"',
        f'readonly REVIEW_STATE_DIR="{state}"',
    ).replace(
        'readonly REVIEW_TEST_ROOT="/tmp/claude-review-tests"',
        f'readonly REVIEW_TEST_ROOT="{review_tests}"',
    ).replace(
        "readonly REVIEW_TEST_TIMEOUT_SECONDS=600",
        f"readonly REVIEW_TEST_TIMEOUT_SECONDS={timeout_seconds}",
    ).replace(
        "readonly REVIEW_TEST_KILL_AFTER_SECONDS=5",
        f"readonly REVIEW_TEST_KILL_AFTER_SECONDS={kill_after_seconds}",
    )
    _write_executable(helper, source)

    shared = state / "envs" / "shared"
    (shared / "bin").mkdir(parents=True)
    (shared / "lib" / "python3.12" / "site-packages" / "pytest").mkdir(
        parents=True
    )
    (shared / "pyvenv.cfg").write_text(
        "home = /usr/local/bin\n"
        "include-system-site-packages = false\n"
        "version = 3.12.0\n"
        "executable = /usr/local/bin/python3.12\n",
        encoding="utf-8",
    )
    (shared / "bin" / "python").symlink_to("/usr/local/bin/python3.12")
    pytest_package = shared / "lib" / "python3.12" / "site-packages" / "pytest"
    (pytest_package / "__init__.py").write_text("", encoding="utf-8")
    (pytest_package / "__main__.py").write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "test_path = next(Path(arg) for arg in sys.argv[1:] if arg.endswith('.py'))\n"
        "namespace = {}\n"
        "exec(compile(test_path.read_text(), str(test_path), 'exec'), namespace)\n"
        "for name, value in sorted(namespace.items()):\n"
        "    if name.startswith('test_') and callable(value):\n"
        "        value()\n",
        encoding="utf-8",
    )
    review_tests.mkdir(mode=0o755)
    (state / "workspace").write_text(f"{REPOSITORY_ROOT}\n", encoding="utf-8")
    (state / "envs" / "selected-projects").write_text("root\n", encoding="utf-8")
    git_mask = state / "git-mask"
    if (REPOSITORY_ROOT / ".git").is_dir():
        git_mask.mkdir()
    else:
        git_mask.touch()
    for path in (state, state / "envs", shared, review_tests):
        path.chmod(0o755)

    environment = os.environ.copy()
    environment["GITHUB_RUN_ID"] = f"pytest{os.getpid()}"
    environment["GITHUB_RUN_ATTEMPT"] = "0"
    return helper, review_tests, environment


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
    assert "dst=/workspace/.git,readonly" in docker_log
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

    assert result.returncode == 64
    assert "not a real directory" in result.stderr


def test_self_test_fails_cleanly_before_prepare(tmp_path: Path) -> None:
    helper, _, _, environment = _make_helper(tmp_path)

    result = _run(helper, environment, "--self-test")

    assert result.returncode == 64
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


def test_dependency_build_temporary_storage_is_disk_backed() -> None:
    source = HELPER.read_text(encoding="utf-8")

    assert '--mount "type=bind,src=${REVIEW_BUILD_ROOT},dst=/review-build"' in source
    assert "--env TMPDIR=/review-build/tmp" in source
    assert "UV_CACHE_DIR" not in source


def test_sandbox_checks_network_interfaces_and_rejects_timeouts() -> None:
    source = HELPER.read_text(encoding="utf-8")

    assert "socket.if_nameindex()" in source
    assert 'interfaces <= {"lo"}' in source
    assert "except TimeoutError as error:" in source
    assert "error.errno in allowed" in source


def test_prepare_and_test_execution_have_wall_clock_limits() -> None:
    source = HELPER.read_text(encoding="utf-8")

    assert '"${REVIEW_PREPARE_TIMEOUT_SECONDS}"' in source
    assert '"${REVIEW_TEST_TIMEOUT_SECONDS}"' in source
    assert 'docker kill "${container_name}"' in source


@pytest.mark.skipif(not _docker_image_is_ready(), reason="local Docker image unavailable")
def test_real_docker_executes_a_review_test_through_the_nested_mount(
    tmp_path: Path,
) -> None:
    helper, review_tests, environment = _make_real_docker_helper(tmp_path)
    probe = review_tests / "test_probe.py"
    probe.write_text(
        "from pathlib import Path\n"
        "def test_git_metadata_is_masked():\n"
        "    metadata = Path('/workspace/.git')\n"
        "    if metadata.is_file():\n"
        "        assert not metadata.read_text()\n"
        "    else:\n"
        "        assert not any(metadata.iterdir())\n",
        encoding="utf-8",
    )

    result = _run(helper, environment, "root", str(probe))

    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(not _docker_image_is_ready(), reason="local Docker image unavailable")
def test_timeout_removes_the_actual_container(tmp_path: Path) -> None:
    helper, review_tests, environment = _make_real_docker_helper(
        tmp_path, timeout_seconds=1, kill_after_seconds=1
    )
    probe = review_tests / "test_sleep.py"
    probe.write_text("import time\ndef test_sleep():\n    time.sleep(60)\n", encoding="utf-8")
    name_filter = f"name=review-falsify-test-{environment['GITHUB_RUN_ID']}-"

    started = time.monotonic()
    result = _run(helper, environment, "root", str(probe))
    elapsed = time.monotonic() - started
    running = subprocess.run(
        ["docker", "ps", "--filter", name_filter, "--format", "{{.Names}}"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert elapsed < 10
    assert not running.stdout.strip(), running.stdout


def test_harness_refusal_does_not_overlap_pytest_statuses() -> None:
    source = HELPER.read_text(encoding="utf-8")

    assert "exit 64" in source


def test_missing_test_path_is_a_usage_error(tmp_path: Path) -> None:
    helper, _, review_tests, environment = _make_helper(tmp_path)
    prepared = _prepare(helper, environment)
    assert prepared.returncode == 0, prepared.stderr

    result = _run(helper, environment, "root", str(review_tests / "missing.py"))

    assert result.returncode == 64
    assert result.stderr.startswith("review-falsify:")
