from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
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
if [[ " $* " == *test_review_integrity_canary_* ]]; then
  canary_path="$(find "${FAKE_REVIEW_TEST_ROOT}" -maxdepth 1 \
    -name 'test_review_integrity_canary_*.py' -print -quit)"
  [[ -n "${canary_path}" && "$(stat -c %a "${canary_path}")" == 644 ]] \
    || exit 87
  exit "${FAKE_TEST_STATUS:-0}"
fi
if [[ " $* " == *" --entrypoint /bin/bash "* ]]; then
  if [[ " $* " == *"dst=/runner-env"* ]]; then
    mkdir -p "${FAKE_REVIEW_ENV_ROOT}/runner/bin"
    printf 'home = /container-only\n' \
      >"${FAKE_REVIEW_ENV_ROOT}/runner/pyvenv.cfg"
    ln -s /container-only/python3.12 \
      "${FAKE_REVIEW_ENV_ROOT}/runner/bin/python"
    exit "${FAKE_RUNNER_PREPARE_STATUS:-0}"
  fi
  if [[ " $* " == *"dst=/project-env"* ]]; then
    mkdir -p \
      "${FAKE_REVIEW_ENV_ROOT}/project/lib/python3.12/site-packages"
    exit "${FAKE_PREPARE_STATUS:-0}"
  fi
fi
exit 0
""",
    )
    _write_executable(
        fake_bin / "chmod",
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "${FAKE_CHMOD_FAIL:-}" == 1 && " $* " == *test_unreadable.py* ]]; then
  exit 1
fi
exec /usr/bin/chmod "$@"
""",
    )
    environment = os.environ.copy()
    environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
    environment["FAKE_REVIEW_ENV_ROOT"] = str(state / "envs")
    environment["FAKE_REVIEW_TEST_ROOT"] = str(review_tests)
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

    runner = state / "envs" / "runner"
    project = state / "envs" / "project"
    (runner / "bin").mkdir(parents=True)
    (runner / "lib" / "python3.12" / "site-packages" / "pytest").mkdir(
        parents=True
    )
    (project / "lib" / "python3.12" / "site-packages").mkdir(parents=True)
    (runner / "pyvenv.cfg").write_text(
        "home = /usr/local/bin\n"
        "include-system-site-packages = false\n"
        "version = 3.12.0\n"
        "executable = /usr/local/bin/python3.12\n",
        encoding="utf-8",
    )
    (runner / "bin" / "python").symlink_to("/usr/local/bin/python3.12")
    pytest_package = runner / "lib" / "python3.12" / "site-packages" / "pytest"
    (pytest_package / "__init__.py").write_text(
        "from pathlib import Path\n"
        "from types import SimpleNamespace\n"
        "def main(arguments, plugins=None):\n"
        "    paths = [Path(arg) for arg in arguments if arg.endswith('.py')]\n"
        "    recorder = plugins[0]\n"
        "    items = []\n"
        "    tests = []\n"
        "    for path in paths:\n"
        "        namespace = {}\n"
        "        exec(compile(path.read_text(), str(path), 'exec'), namespace)\n"
        "        for name, value in sorted(namespace.items()):\n"
        "            if name.startswith('test_') and callable(value):\n"
        "                nodeid = f'{path}::{name}'\n"
        "                items.append(SimpleNamespace(nodeid=nodeid, path=path))\n"
        "                tests.append((nodeid, value))\n"
        "    recorder.pytest_collection_modifyitems(None, None, items)\n"
        "    failed = False\n"
        "    for nodeid, value in tests:\n"
        "        outcome = 'passed'\n"
        "        try:\n"
        "            value()\n"
        "        except AssertionError:\n"
        "            outcome = 'failed'\n"
        "            failed = True\n"
        "        recorder.pytest_runtest_logreport(SimpleNamespace(\n"
        "            nodeid=nodeid, when='call', outcome=outcome,\n"
        "            failed=outcome == 'failed',\n"
        "            passed=outcome == 'passed'))\n"
        "    return 1 if failed else 0\n",
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
    for path in (state, state / "envs", runner, project, review_tests):
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


def test_failed_prepare_cannot_leave_a_runnable_partial_environment(
    tmp_path: Path,
) -> None:
    helper, state, review_tests, environment = _make_helper(tmp_path)
    environment["FAKE_PREPARE_STATUS"] = "1"

    prepared = _prepare(helper, environment)
    probe = review_tests / "test_probe.py"
    probe.write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    attempted_run = _run(helper, environment, "root", str(probe))

    assert prepared.returncode == 1
    assert attempted_run.returncode == 64
    assert "sandbox is not prepared" in attempted_run.stderr
    assert not (state / "workspace").exists()
    assert not (state / "envs").exists()


def test_rejected_prepare_invalidates_the_previous_environment(tmp_path: Path) -> None:
    helper, _, review_tests, environment = _make_helper(tmp_path)
    prepared = _prepare(helper, environment)
    assert prepared.returncode == 0, prepared.stderr

    rejected = _run(helper, environment, "--prepare", str(REPOSITORY_ROOT), "0" * 40)
    probe = review_tests / "test_probe.py"
    probe.write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    attempted_run = _run(helper, environment, "root", str(probe))

    assert rejected.returncode == 64
    assert attempted_run.returncode == 64
    assert "sandbox is not prepared" in attempted_run.stderr


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


def test_self_test_rejects_a_missing_review_test_root(tmp_path: Path) -> None:
    helper, _, review_tests, environment = _make_helper(tmp_path)
    prepared = _prepare(helper, environment)
    assert prepared.returncode == 0, prepared.stderr
    review_tests.rmdir()

    result = _run(helper, environment, "--self-test")

    assert result.returncode == 64
    assert "review test root" in result.stderr


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


def test_pytest_integrity_controls_are_part_of_every_sandbox_run() -> None:
    source = HELPER.read_text(encoding="utf-8")

    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in source
    assert '-c /dev/null' in source
    assert '--confcutdir "${test_mount}"' in source
    assert '--rootdir "${test_mount}"' in source
    assert "test_review_integrity_canary_must_fail" in source
    assert 'recorder.canary_calls != ["failed"]' in source
    assert "completed.returncode != 0" in source
    assert "pass_fds=(write_fd,)" in source
    assert "recorder.user_passed != recorder.user_items" in source
    assert 'chmod 0644 -- "${canary_file}"' in source
    assert 'trap \'rm -f -- "${canary_file}"\' EXIT' in source


def test_trusted_pytest_runner_is_separate_from_pr_dependency_environment() -> None:
    source = HELPER.read_text(encoding="utf-8")

    assert "dst=/runner-env" in source
    assert "dst=/project-env" in source
    assert "/review-envs/runner/bin/python" in source
    assert "/review-envs/project/${REVIEW_SITE_PACKAGES}" in source
    runner_section, project_section = source.split(
        "prepare_container_name=\"review-falsify-project-", maxsplit=1
    )
    assert "dst=/runner-env" in runner_section
    assert "dst=/runner-env" not in project_section


def _supervisor_program() -> str:
    source = HELPER.read_text(encoding="utf-8")
    start = source.index("-I -c '") + len("-I -c '")
    end = source.index("\n' \"${project_path}\"", start)
    return source[start:end]


def _run_inner_wrapper(
    tmp_path: Path, user_source: str
) -> subprocess.CompletedProcess[str]:
    workspace = tmp_path / "workspace"
    test_mount = workspace / "tests" / ".review-tests"
    test_mount.mkdir(parents=True)
    canary = test_mount / "test_review_integrity_canary_7_31337.py"
    canary.write_text(
        "def test_review_integrity_canary_must_fail():\n"
        "    assert False, 'review integrity canary'\n",
        encoding="utf-8",
    )
    probe = test_mount / "test_probe.py"
    probe.write_text(user_source, encoding="utf-8")
    environment = os.environ.copy()
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    return subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            _supervisor_program(),
            str(workspace),
            str(workspace),
            str(canary),
            str(probe),
            "-c",
            "/dev/null",
            "--confcutdir",
            str(test_mount),
            "--rootdir",
            str(test_mount),
            "-q",
            "--disable-warnings",
        ],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )


def test_real_pytest_wrapper_distinguishes_passing_and_failing_review_tests(
    tmp_path: Path,
) -> None:
    passing = _run_inner_wrapper(
        tmp_path / "passing", "def test_ok():\n    assert True\n"
    )
    failing = _run_inner_wrapper(
        tmp_path / "failing", "def test_counterexample():\n    assert False\n"
    )

    assert passing.returncode == 0, passing.stdout + passing.stderr
    assert failing.returncode == 1, failing.stdout + failing.stderr


def test_real_pytest_wrapper_classifies_fixture_errors_as_malformed(
    tmp_path: Path,
) -> None:
    result = _run_inner_wrapper(
        tmp_path,
        "import pytest\n"
        "@pytest.fixture\n"
        "def broken_fixture():\n"
        "    raise RuntimeError('fixture failed')\n"
        "def test_uses_fixture(broken_fixture):\n"
        "    assert True\n",
    )

    assert result.returncode == 2, result.stdout + result.stderr


def test_direct_os_exit_cannot_forge_a_passing_review_result(tmp_path: Path) -> None:
    result = _run_inner_wrapper(
        tmp_path,
        "import os\n"
        "def test_forge_encoded_pass():\n"
        "    os._exit(80)\n",
    )

    assert result.returncode == 64, result.stdout + result.stderr


@pytest.mark.parametrize(
    "user_source",
    [
        "import pytest\n@pytest.mark.skip\ndef test_skipped():\n    pass\n",
        "import pytest\n@pytest.mark.xfail\ndef test_expected_failure():\n"
        "    assert False\n",
    ],
)
def test_skipped_or_xfailed_review_tests_are_not_accepted_as_passing(
    tmp_path: Path, user_source: str
) -> None:
    result = _run_inner_wrapper(tmp_path, user_source)

    assert result.returncode == 5, result.stdout + result.stderr


def test_canary_is_readable_even_when_the_reviewer_uses_a_restrictive_umask(
    tmp_path: Path,
) -> None:
    helper, _, review_tests, environment = _make_helper(tmp_path)
    prepared = _prepare(helper, environment)
    assert prepared.returncode == 0, prepared.stderr
    probe = review_tests / "test_probe.py"
    probe.write_text("def test_ok():\n    assert True\n", encoding="utf-8")

    result = subprocess.run(
        [
            "bash",
            "-c",
            'umask 077; exec "$@"',
            "review-test",
            str(helper),
            "root",
            str(probe),
        ],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(not _docker_image_is_ready(), reason="local Docker image unavailable")
def test_pr_atexit_handler_cannot_turn_a_failing_review_test_into_a_pass(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    (workspace / ".git").mkdir(parents=True)
    (workspace / "tests" / ".review-tests").mkdir(parents=True)
    (workspace / "typo_helper.py").write_text(
        "import atexit\n"
        "import os\n"
        "def normalise(text):\n"
        "    return text\n"
        "@atexit.register\n"
        "def _flush():\n"
        "    os._exit(0)\n",
        encoding="utf-8",
    )
    helper, review_tests, environment = _make_real_docker_helper(tmp_path)
    state = tmp_path / "real-state"
    (state / "workspace").write_text(f"{workspace}\n", encoding="utf-8")
    (state / "git-mask").unlink()
    (state / "git-mask").mkdir()
    probe = review_tests / "test_falsify.py"
    probe.write_text(
        "from typo_helper import normalise\n"
        "def test_counterexample():\n"
        "    assert normalise('wrong') == 'right'\n",
        encoding="utf-8",
    )

    result = _run(helper, environment, "root", str(probe))

    assert result.returncode == 1, result.stderr


def test_missing_test_path_is_a_usage_error(tmp_path: Path) -> None:
    helper, _, review_tests, environment = _make_helper(tmp_path)
    prepared = _prepare(helper, environment)
    assert prepared.returncode == 0, prepared.stderr

    result = _run(helper, environment, "root", str(review_tests / "missing.py"))

    assert result.returncode == 64
    assert result.stderr.startswith("review-falsify:")


def test_chmod_setup_failure_is_a_harness_refusal(tmp_path: Path) -> None:
    helper, _, review_tests, environment = _make_helper(tmp_path)
    prepared = _prepare(helper, environment)
    assert prepared.returncode == 0, prepared.stderr
    probe = review_tests / "test_unreadable.py"
    probe.write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    environment["FAKE_CHMOD_FAIL"] = "1"

    result = _run(helper, environment, "root", str(probe))

    assert result.returncode == 64
    assert "could not be made sandbox-readable" in result.stderr
    assert not list(review_tests.glob("test_review_integrity_canary_*.py"))


def test_nested_review_test_is_rejected_before_the_sandbox(tmp_path: Path) -> None:
    helper, _, review_tests, environment = _make_helper(tmp_path)
    prepared = _prepare(helper, environment)
    assert prepared.returncode == 0, prepared.stderr
    nested = review_tests / "case"
    nested.mkdir(mode=0o700)
    probe = nested / "test_probe.py"
    probe.write_text("def test_ok():\n    assert True\n", encoding="utf-8")

    result = _run(helper, environment, "root", str(probe))

    assert result.returncode == 64
    assert "directly below" in result.stderr
