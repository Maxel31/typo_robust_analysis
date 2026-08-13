#!/usr/bin/env bash
set -euo pipefail

readonly REVIEW_IMAGE="ghcr.io/astral-sh/uv:python3.12-bookworm"
readonly REVIEW_PYTEST_VERSION="9.0.3"
readonly REVIEW_SITE_PACKAGES="lib/python3.12/site-packages"
readonly REVIEW_STATE_DIR="/var/lib/claude-pr-review"
readonly REVIEW_WORKSPACE_FILE="${REVIEW_STATE_DIR}/workspace"
readonly REVIEW_ENV_ROOT="${REVIEW_STATE_DIR}/envs"
readonly REVIEW_BUILD_ROOT="${REVIEW_STATE_DIR}/build"
readonly REVIEW_GIT_MASK="${REVIEW_STATE_DIR}/git-mask"
readonly REVIEW_TEST_ROOT="/tmp/claude-review-tests"
readonly REVIEW_PREPARE_TIMEOUT_SECONDS=1800
readonly REVIEW_TEST_TIMEOUT_SECONDS=600
readonly REVIEW_PREPARE_KILL_AFTER_SECONDS=30
readonly REVIEW_TEST_KILL_AFTER_SECONDS=5

die() {
  printf 'review-falsify: %s\n' "$*" >&2
  # pytest reserves statuses 1-5. Keep harness refusal distinguishable from
  # both a falsifying test failure and a malformed review test.
  exit 64
}

require_prepared_workspace() {
  [[ -r "${REVIEW_WORKSPACE_FILE}" ]] || die "sandbox is not prepared"
  local workspace
  workspace="$(<"${REVIEW_WORKSPACE_FILE}")"
  [[ -e "${workspace}/.git" ]] || die "prepared workspace is unavailable"
  printf '%s\n' "${workspace}"
}

run_docker_with_timeout() {
  [[ $# -ge 4 ]] || die "internal error: incomplete Docker timeout invocation"
  local timeout_seconds="$1"
  local kill_after_seconds="$2"
  local container_name="$3"
  shift 3

  local status=0
  timeout --signal=TERM --kill-after="${kill_after_seconds}s" \
    "${timeout_seconds}s" docker run --name "${container_name}" "$@" \
    || status=$?
  if ((status != 0)); then
    # GNU timeout only owns the Docker client. A container PID 1 may ignore
    # the proxied TERM and outlive that client, so remove it explicitly.
    timeout --signal=KILL 30s docker kill "${container_name}" >/dev/null 2>&1 \
      || true
  fi
  timeout --signal=KILL 30s docker rm -f "${container_name}" >/dev/null 2>&1 \
    || true
  return "${status}"
}

prepare_sandbox() {
  # Any prepare attempt invalidates the previous environment before validating
  # its new inputs. Otherwise a rejected retry could silently leave stale
  # dependencies runnable on a persistent runner.
  sudo install -d -m 0755 -o root -g root "${REVIEW_STATE_DIR}"
  sudo rm -f -- "${REVIEW_WORKSPACE_FILE}"
  [[ $# -eq 2 ]] || die "usage: review-falsify --prepare WORKSPACE BASE_SHA"
  local base_sha changed_paths workspace
  if ! workspace="$(realpath --canonicalize-existing -- "$1" 2>/dev/null)"; then
    die "WORKSPACE must be an existing Git checkout"
  fi
  [[ -e "${workspace}/.git" ]] || die "WORKSPACE must be a Git checkout"
  base_sha="$2"
  [[ "${base_sha}" =~ ^[0-9a-f]{40}$ ]] || die "BASE_SHA must be a full commit digest"
  git -C "${workspace}" cat-file -e "${base_sha}^{commit}" \
    || die "BASE_SHA is not present in the checkout"
  if ! changed_paths="$(
    git -C "${workspace}" diff --name-only "${base_sha}...HEAD"
  )"; then
    die "BASE_SHA does not share usable history with the checkout"
  fi

  # A previous run makes the environment root-owned and immutable to the
  # runner. Rebuild it from scratch so retries cannot mix dependency states.
  sudo rm -rf -- "${REVIEW_ENV_ROOT}" "${REVIEW_BUILD_ROOT}" "${REVIEW_GIT_MASK}"
  sudo install -d -m 0755 -o "$(id -u)" -g "$(id -g)" "${REVIEW_ENV_ROOT}"
  install -d -m 0755 "${REVIEW_ENV_ROOT}/runner"
  install -d -m 0755 "${REVIEW_ENV_ROOT}/project"
  sudo install -d -m 0755 -o "$(id -u)" -g "$(id -g)" "${REVIEW_BUILD_ROOT}"
  install -d -m 0755 "${REVIEW_BUILD_ROOT}/tmp"
  if [[ -d "${workspace}/.git" ]]; then
    sudo install -d -m 0555 -o root -g root "${REVIEW_GIT_MASK}"
  else
    sudo install -m 0444 -o root -g root /dev/null "${REVIEW_GIT_MASK}"
  fi
  # The reviewer owns this directory; the sandbox only needs read/execute.
  install -d -m 0755 "${REVIEW_TEST_ROOT}"

  docker pull --quiet "${REVIEW_IMAGE}" >/dev/null
  local -a selected_projects=(root)
  if [[ -f "${workspace}/projects/typo-cot/pyproject.toml" ]] \
    && grep -Eq '^(pyproject\.toml|uv\.lock|projects/typo-cot/)' \
      <<<"${changed_paths}"; then
    selected_projects+=(typo-cot)
  fi
  if [[ -f "${workspace}/projects/typo-robust-training/pyproject.toml" ]] \
    && grep -Eq '^(pyproject\.toml|uv\.lock|projects/typo-robust-training/)' \
      <<<"${changed_paths}"; then
    selected_projects+=(typo-robust-training)
  fi
  printf '%s\n' "${selected_projects[@]}" >"${REVIEW_ENV_ROOT}/selected-projects"
  cp -- "${REVIEW_ENV_ROOT}/selected-projects" "${REVIEW_BUILD_ROOT}/selected-projects"

  local prepare_status=0
  local prepare_container_name
  prepare_container_name="review-falsify-runner-${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-0}-$$-${RANDOM}"
  run_docker_with_timeout \
    "${REVIEW_PREPARE_TIMEOUT_SECONDS}" \
    "${REVIEW_PREPARE_KILL_AFTER_SECONDS}" \
    "${prepare_container_name}" \
    --rm \
    --network bridge \
    --cap-drop ALL \
    --security-opt no-new-privileges \
    --pids-limit 512 \
    --memory 10g \
    --cpus 4 \
    --user "$(id -u):$(id -g)" \
    --tmpfs /tmp:rw,nosuid,nodev,size=1g \
    --mount "type=bind,src=${REVIEW_ENV_ROOT}/runner,dst=/runner-env" \
    --workdir / \
    --env HOME=/tmp \
    --entrypoint /bin/bash \
    "${REVIEW_IMAGE}" \
    -euo pipefail -c "
      uv venv --python /usr/local/bin/python3.12 /runner-env
      uv pip install --python /runner-env/bin/python --only-binary :all: \
        pytest==${REVIEW_PYTEST_VERSION} --no-cache
    " || prepare_status=$?
  if ((prepare_status != 0)); then
    sudo rm -rf -- "${REVIEW_ENV_ROOT}" "${REVIEW_BUILD_ROOT}" "${REVIEW_GIT_MASK}"
    return "${prepare_status}"
  fi
  # The PR-controlled dependency build below can write only its sibling mount,
  # never the trusted pytest runner used to interpret review results.
  sudo chown -R root:root "${REVIEW_ENV_ROOT}/runner"
  sudo chmod -R go-w "${REVIEW_ENV_ROOT}/runner"

  # PR-controlled package build hooks run online but without host credentials.
  # Their only writable persistent mount is /project-env; the trusted runner is
  # neither mounted nor otherwise reachable from this preparation container.
  prepare_container_name="review-falsify-project-${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-0}-$$-${RANDOM}"
  run_docker_with_timeout \
    "${REVIEW_PREPARE_TIMEOUT_SECONDS}" \
    "${REVIEW_PREPARE_KILL_AFTER_SECONDS}" \
    "${prepare_container_name}" \
    --rm \
    --network bridge \
    --cap-drop ALL \
    --security-opt no-new-privileges \
    --pids-limit 512 \
    --memory 10g \
    --cpus 4 \
    --user "$(id -u):$(id -g)" \
    --tmpfs /tmp:rw,nosuid,nodev,size=1g \
    --mount "type=bind,src=${workspace},dst=/workspace,readonly" \
    --mount "type=bind,src=${REVIEW_GIT_MASK},dst=/workspace/.git,readonly" \
    --mount "type=bind,src=${REVIEW_ENV_ROOT}/project,dst=/project-env" \
    --mount "type=bind,src=${REVIEW_BUILD_ROOT},dst=/review-build" \
    --workdir /workspace \
    --env HOME=/tmp \
    --env TMPDIR=/review-build/tmp \
    --entrypoint /bin/bash \
    "${REVIEW_IMAGE}" \
    -euo pipefail -c '
      UV_PROJECT_ENVIRONMENT=/project-env \
        uv sync --locked --dev --no-install-workspace --no-cache
      while IFS= read -r project; do
        if [[ "${project}" != root ]]; then
          UV_PROJECT_ENVIRONMENT=/project-env \
            uv sync --project "projects/${project}" --locked --dev --all-extras \
              --no-install-project --inexact --no-cache
        fi
      done </review-build/selected-projects
    ' || prepare_status=$?
  sudo rm -rf -- "${REVIEW_BUILD_ROOT}"
  if ((prepare_status != 0)); then
    sudo rm -rf -- "${REVIEW_ENV_ROOT}" "${REVIEW_GIT_MASK}"
    return "${prepare_status}"
  fi
  sudo chown -R root:root "${REVIEW_ENV_ROOT}"
  sudo chmod -R go-w "${REVIEW_ENV_ROOT}"
  # This file is the completion marker and must be the final prepare write.
  printf '%s\n' "${workspace}" | sudo tee "${REVIEW_WORKSPACE_FILE}" >/dev/null
  sudo chmod 0444 "${REVIEW_WORKSPACE_FILE}"
}

execute_sandboxed_pytest() {
  [[ $# -eq 10 ]] || die "internal error: incomplete pytest invocation"
  local workspace="$1"
  local python_path="$2"
  local project_path="$3"
  local dependency_path="$4"
  local test_mount="$5"
  local workdir="$6"
  local test_file="$7"
  local test_relative="$8"
  local canary_file="$9"
  local canary_relative="${10}"

  chmod 0644 -- "${test_file}" \
    || die "TEST_FILE could not be made sandbox-readable"
  chmod 0644 -- "${canary_file}" \
    || die "review integrity canary could not be made sandbox-readable"
  local test_container_name
  test_container_name="review-falsify-test-${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-0}-$$-${RANDOM}"
  local sandbox_status=0
  # Repository pytest configuration and conftest files are intentionally
  # excluded: only the reviewer-authored test and imported product code run.
  run_docker_with_timeout \
    "${REVIEW_TEST_TIMEOUT_SECONDS}" \
    "${REVIEW_TEST_KILL_AFTER_SECONDS}" \
    "${test_container_name}" \
    --rm --pull never \
    --network none \
    --read-only \
    --cap-drop ALL \
    --security-opt no-new-privileges \
    --pids-limit 256 \
    --memory 8g \
    --cpus 4 \
    --user 65534:65534 \
    --tmpfs /tmp:rw,nosuid,nodev,size=1g \
    --mount "type=bind,src=${workspace},dst=/workspace,readonly" \
    --mount "type=bind,src=${REVIEW_GIT_MASK},dst=/workspace/.git,readonly" \
    --mount "type=bind,src=${REVIEW_ENV_ROOT},dst=/review-envs,readonly" \
    --mount "type=bind,src=${REVIEW_TEST_ROOT},dst=${test_mount},readonly" \
    --workdir "${workdir}" \
    --env HOME=/tmp \
    --env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
    --entrypoint "${python_path}" \
    "${REVIEW_IMAGE}" \
    -I -c '
import os
import hashlib
import hmac
import secrets
import subprocess
import sys

# Keep result interpretation in this trusted parent. The child imports and
# executes PR code, but a normal exit is insufficient: it must complete a
# parent-initiated challenge after pytest returns. This catches accidental and
# obvious early exits; it is not a security proof against arbitrary hostile
# Python sharing the child process.
CHILD = r"""
import hashlib
import hmac
import os
import site
import sys
import pytest

def _trusted_run():
    trusted_exit = os._exit
    trusted_read = os.read
    trusted_write = os.write
    project_path = sys.argv[1]
    dependency_path = sys.argv[2]
    canary_path = sys.argv[3]
    test_path = sys.argv[4]
    result_fd = int(sys.argv[5])
    challenge_fd = int(sys.argv[6])
    pytest_args = sys.argv[7:]
    # Do not expose protocol descriptors or pytest paths through sys.argv to
    # imported PR modules. Remove this controller entry point as well.
    sys.argv[:] = [sys.argv[0]]
    globals().pop("_trusted_run", None)

    class ResultRecorder:
        def __init__(self, expected_canary_path):
            self.canary_path = os.path.realpath(expected_canary_path)
            self.canary_nodeids = set()
            self.canary_calls = []
            self.xfail_nodeids = set()
            self.user_items = 0
            self.user_passed = 0
            self.user_failed = False
            self.user_error = False
            self.collection_failed = False

        def _is_canary(self, nodeid):
            return nodeid in self.canary_nodeids

        def pytest_collection_modifyitems(self, session, config, items):
            for item in items:
                if os.path.realpath(str(item.path)) == self.canary_path:
                    self.canary_nodeids.add(item.nodeid)
                elif hasattr(item, "iter_markers") and any(
                    item.iter_markers(name="xfail")
                ):
                    self.xfail_nodeids.add(item.nodeid)
            self.user_items = sum(
                not self._is_canary(item.nodeid) for item in items
            )

        def pytest_collectreport(self, report):
            if report.failed:
                self.collection_failed = True

        def pytest_runtest_logreport(self, report):
            if self._is_canary(report.nodeid):
                if report.when == "call":
                    self.canary_calls.append(report.outcome)
                return
            # Neither an expected failure nor a strict XPASS is evidence that
            # a reviewer-authored counterexample reproduced a product defect.
            if report.nodeid in self.xfail_nodeids:
                return
            if report.failed and report.when == "call":
                self.user_failed = True
            elif report.failed:
                self.user_error = True
            elif (report.when == "call" and report.passed
                  and not hasattr(report, "wasxfail")):
                self.user_passed += 1

    # pytest is already loaded from the root-owned runner. Only then expose the
    # read-only PR source and process the separately built dependency
    # environment like a normal venv, including its .pth files.
    sys.path.append(project_path)
    site.addsitedir(dependency_path)
    sys.dont_write_bytecode = True
    recorder = ResultRecorder(canary_path)
    try:
        pytest_status = int(pytest.main(
            [*pytest_args, canary_path, test_path], plugins=[recorder]
        ))
    except BaseException:
        kind = "internal"
    else:
        if pytest_status == int(pytest.ExitCode.INTERNAL_ERROR):
            kind = "internal"
        elif pytest_status == int(pytest.ExitCode.USAGE_ERROR):
            kind = "usage"
        elif recorder.collection_failed:
            kind = "malformed"
        elif recorder.canary_calls != ["failed"]:
            kind = "integrity"
        # Once a call-phase assertion falsifies the feature, a later teardown
        # error must not erase that evidence.
        elif recorder.user_failed:
            kind = "failed"
        elif recorder.user_error:
            kind = "malformed"
        elif pytest_status == int(pytest.ExitCode.INTERRUPTED):
            kind = "interrupted"
        elif (
            recorder.user_items == 0
            or pytest_status == int(pytest.ExitCode.NO_TESTS_COLLECTED)
        ):
            kind = "empty"
        elif recorder.user_passed != recorder.user_items:
            kind = "empty"
        else:
            kind = "passed"

    trusted_write(result_fd, f"ready:{kind}\n".encode("ascii"))
    challenge = trusted_read(challenge_fd, 32)
    if len(challenge) != 32:
        trusted_exit(64)
    proof = hmac.new(challenge, kind.encode("ascii"), hashlib.sha256).hexdigest()
    trusted_write(result_fd, f"done:{proof}\n".encode("ascii"))
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except BaseException:
        pass
    trusted_exit(0)

_trusted_run()
"""

project_path = sys.argv[1]
dependency_path = sys.argv[2]
canary_path = sys.argv[3]
test_path = sys.argv[4]
pytest_args = sys.argv[5:]
result_read_fd, result_write_fd = os.pipe()
challenge_read_fd, challenge_write_fd = os.pipe()
completed = subprocess.Popen(
    [sys.executable, "-I", "-c", CHILD, project_path, dependency_path,
     canary_path, test_path, str(result_write_fd), str(challenge_read_fd),
     *pytest_args],
    pass_fds=(result_write_fd, challenge_read_fd),
)
statuses = {"passed": 0, "failed": 1, "interrupted": 2, "internal": 3,
            "usage": 4, "malformed": 4, "empty": 5, "integrity": 64}
os.close(result_write_fd)
os.close(challenge_read_fd)
protocol_ok = True
kind = ""
challenge = secrets.token_bytes(32)
with os.fdopen(result_read_fd, "rb", buffering=0) as result_stream:
    ready = result_stream.readline(256)
    if ready.startswith(b"ready:") and ready.endswith(b"\n"):
        try:
            kind = ready[6:-1].decode("ascii")
        except UnicodeDecodeError:
            protocol_ok = False
    else:
        protocol_ok = False
    if kind not in statuses:
        protocol_ok = False
    if protocol_ok:
        try:
            os.write(challenge_write_fd, challenge)
        except BrokenPipeError:
            protocol_ok = False
    os.close(challenge_write_fd)
    done = result_stream.readline(256)
returncode = completed.wait()
expected = b"done:" + hmac.new(
    challenge, kind.encode("ascii"), hashlib.sha256
).hexdigest().encode("ascii") + b"\n"
if returncode != 0 or not protocol_ok or done != expected:
    raise SystemExit(64)
raise SystemExit(statuses[kind])
' "${project_path}" "${dependency_path}" \
    "${test_mount}/${canary_relative}" \
    "${test_mount}/${test_relative}" \
    -c /dev/null \
    --confcutdir "${test_mount}" \
    --rootdir "${test_mount}" \
    -q --disable-warnings \
    || sandbox_status=$?
  return "${sandbox_status}"
}

run_test() {
  [[ $# -eq 2 ]] || die "usage: review-falsify {root|typo-cot|typo-robust-training} TEST_FILE"
  local project="$1"
  local dependency_path host_test_mount python_path pythonpath test_file test_mount test_relative workdir workspace
  workspace="$(require_prepared_workspace)"
  if ! test_file="$(realpath --canonicalize-existing -- "$2" 2>/dev/null)"; then
    die "TEST_FILE must be an existing Python file below ${REVIEW_TEST_ROOT}"
  fi
  [[ -f "${test_file}" && "${test_file}" == "${REVIEW_TEST_ROOT}/"*.py ]] \
    || die "TEST_FILE must be an existing Python file below ${REVIEW_TEST_ROOT}"
  test_relative="${test_file#${REVIEW_TEST_ROOT}/}"
  [[ "${test_relative}" != */* ]] \
    || die "TEST_FILE must be directly below ${REVIEW_TEST_ROOT}"

  case "${project}" in
    root)
      python_path="/review-envs/runner/bin/python"
      dependency_path="/review-envs/project/${REVIEW_SITE_PACKAGES}"
      pythonpath="/workspace"
      host_test_mount="${workspace}/tests/.review-tests"
      test_mount="/workspace/tests/.review-tests"
      workdir="/workspace"
      ;;
    typo-cot | typo-robust-training)
      grep -Fxq -- "${project}" "${REVIEW_ENV_ROOT}/selected-projects" \
        || die "the ${project} environment was not provisioned because that project is unchanged"
      python_path="/review-envs/runner/bin/python"
      dependency_path="/review-envs/project/${REVIEW_SITE_PACKAGES}"
      pythonpath="/workspace/projects/${project}/src"
      host_test_mount="${workspace}/projects/${project}/tests/.review-tests"
      test_mount="/workspace/projects/${project}/tests/.review-tests"
      workdir="/workspace/projects/${project}"
      ;;
    *)
      die "unknown project: ${project}"
      ;;
  esac

  # Docker cannot create a nested bind target below the read-only workspace.
  # Require the tracked placeholder and reject PR-controlled symlink redirects.
  local resolved_host_test_mount
  if ! resolved_host_test_mount="$(
    realpath --canonicalize-existing -- "${host_test_mount}" 2>/dev/null
  )"; then
    die "the tracked review-test mount point is missing for ${project}"
  fi
  [[ -d "${host_test_mount}" && "${resolved_host_test_mount}" == "${host_test_mount}" ]] \
    || die "the review-test mount point is not a real directory for ${project}"

  local host_python="${REVIEW_ENV_ROOT}${python_path#/review-envs}"
  if [[ ! -f "${REVIEW_ENV_ROOT}/runner/pyvenv.cfg" ]] \
    || { [[ ! -L "${host_python}" ]] && [[ ! -f "${host_python}" ]]; } \
    || [[ ! -d "${REVIEW_ENV_ROOT}/project/${REVIEW_SITE_PACKAGES}" ]]; then
    die "the ${project} review environment is not installed"
  fi
  # Run the guaranteed failure and reviewer test in the same child pytest
  # session. A separate trusted parent validates the child result report.
  local canary_file canary_relative canary_status=0
  canary_relative="test_review_integrity_canary_$$_${RANDOM}.py"
  canary_file="${REVIEW_TEST_ROOT}/${canary_relative}"
  trap 'rm -f -- "${canary_file}"' EXIT
  if ! printf '%s\n' \
    'def test_review_integrity_canary_must_fail():' \
    '    assert False, "review integrity canary"' \
    >"${canary_file}"; then
    die "review integrity canary could not be created"
  fi
  execute_sandboxed_pytest \
    "${workspace}" "${python_path}" "${pythonpath}" "${dependency_path}" \
    "${test_mount}" \
    "${workdir}" "${test_file}" "${test_relative}" \
    "${canary_file}" "${canary_relative}" \
    || canary_status=$?
  rm -f -- "${canary_file}" \
    || die "review integrity canary could not be removed"
  trap - EXIT
  case "${canary_status}" in
    0 | 1 | 2 | 3 | 4 | 5) return "${canary_status}" ;;
    64) die "pytest result integrity could not be established" ;;
    124 | 137) die "review test exceeded its wall-clock limit" ;;
    125 | 126 | 127) die "review test sandbox could not be started" ;;
    *) die "review test sandbox failed with status ${canary_status}" ;;
  esac
}

self_test() {
  [[ $# -eq 0 ]] || die "usage: review-falsify --self-test"
  require_prepared_workspace >/dev/null
  [[ -d "${REVIEW_TEST_ROOT}" && -w "${REVIEW_TEST_ROOT}" ]] \
    || die "review test root ${REVIEW_TEST_ROOT} is unavailable"
  local test_file="${REVIEW_TEST_ROOT}/test_review_sandbox.py"
  printf '%s\n' \
    'import os' \
    'import errno' \
    'import socket' \
    'from pathlib import Path' \
    '' \
    'def test_review_code_is_uncredentialed_offline_and_read_only():' \
    '    forbidden = ("CLAUDE_CODE_OAUTH_TOKEN", "GITHUB_TOKEN", "ACTIONS_ID_TOKEN_REQUEST_TOKEN", "REVIEW_SANDBOX_SENTINEL")' \
    '    assert all(name not in os.environ for name in forbidden)' \
    '    git_metadata = Path("/workspace/.git")' \
    '    if git_metadata.is_file():' \
    '        assert not git_metadata.read_text(encoding="utf-8")' \
    '    else:' \
    '        assert not any(git_metadata.iterdir())' \
    '    interfaces = {name for _index, name in socket.if_nameindex()}' \
    '    assert interfaces <= {"lo"}, f"unexpected network interfaces: {interfaces}"' \
    '    try:' \
    '        Path("/workspace/.review-write-probe").write_text("unsafe", encoding="utf-8")' \
    '    except OSError:' \
    '        pass' \
    '    else:' \
    '        raise AssertionError("review workspace is writable")' \
    '    probe = socket.socket()' \
    '    probe.settimeout(2.0)' \
    '    try:' \
    '        probe.connect(("1.1.1.1", 53))' \
    '    except TimeoutError as error:' \
    '        raise AssertionError("network probe timed out; isolation unproven") from error' \
    '    except OSError as error:' \
    '        allowed = (errno.ENETUNREACH, errno.EHOSTUNREACH, errno.EPERM, errno.EACCES)' \
    '        assert error.errno in allowed, error' \
    '    else:' \
    '        raise AssertionError("review sandbox has network access")' \
    '    finally:' \
    '        probe.close()' \
    >"${test_file}"
  local test_status=0
  (
    export REVIEW_SANDBOX_SENTINEL=armed
    run_test root "${test_file}"
  ) || test_status=$?
  rm -f -- "${test_file}"
  return "${test_status}"
}

case "${1:-}" in
  --prepare)
    shift
    prepare_sandbox "$@"
    ;;
  --self-test)
    shift
    self_test "$@"
    ;;
  root | typo-cot | typo-robust-training)
    project="$1"
    shift
    run_test "${project}" "$@"
    ;;
  *)
    die "usage: review-falsify --prepare WORKSPACE BASE_SHA | --self-test | {root|typo-cot|typo-robust-training} TEST_FILE"
    ;;
esac
