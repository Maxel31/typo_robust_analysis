#!/usr/bin/env bash
set -euo pipefail

readonly REVIEW_IMAGE="ghcr.io/astral-sh/uv:python3.12-bookworm"
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
  changed_paths="$(git -C "${workspace}" diff --name-only "${base_sha}...HEAD")"

  sudo install -d -m 0755 -o root -g root "${REVIEW_STATE_DIR}"
  # A previous run makes the environment root-owned and immutable to the
  # runner. Rebuild it from scratch so retries cannot mix dependency states.
  sudo rm -rf -- "${REVIEW_ENV_ROOT}" "${REVIEW_BUILD_ROOT}" "${REVIEW_GIT_MASK}"
  sudo install -d -m 0755 -o "$(id -u)" -g "$(id -g)" "${REVIEW_ENV_ROOT}"
  sudo install -d -m 0755 -o "$(id -u)" -g "$(id -g)" "${REVIEW_BUILD_ROOT}"
  install -d -m 0755 "${REVIEW_BUILD_ROOT}/tmp"
  if [[ -d "${workspace}/.git" ]]; then
    sudo install -d -m 0555 -o root -g root "${REVIEW_GIT_MASK}"
  else
    sudo install -m 0444 -o root -g root /dev/null "${REVIEW_GIT_MASK}"
  fi
  printf '%s\n' "${workspace}" | sudo tee "${REVIEW_WORKSPACE_FILE}" >/dev/null
  sudo chmod 0444 "${REVIEW_WORKSPACE_FILE}"
  # The reviewer owns this directory; the sandbox only needs read/execute.
  install -d -m 0755 "${REVIEW_TEST_ROOT}"

  docker pull --quiet "${REVIEW_IMAGE}" >/dev/null
  local -a selected_projects=(root)
  if grep -Eq '^(pyproject\.toml|uv\.lock|projects/typo-cot/)' <<<"${changed_paths}"; then
    selected_projects+=(typo-cot)
  fi
  if [[ -f "${workspace}/projects/typo-robust-training/pyproject.toml" ]] \
    && grep -Eq '^(pyproject\.toml|uv\.lock|projects/typo-robust-training/)' \
      <<<"${changed_paths}"; then
    selected_projects+=(typo-robust-training)
  fi
  printf '%s\n' "${selected_projects[@]}" >"${REVIEW_ENV_ROOT}/selected-projects"

  local prepare_status=0
  local prepare_container_name
  prepare_container_name="review-falsify-prepare-${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-0}-$$-${RANDOM}"
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
    --mount "type=bind,src=${REVIEW_ENV_ROOT},dst=/review-envs" \
    --mount "type=bind,src=${REVIEW_BUILD_ROOT},dst=/review-build" \
    --workdir /workspace \
    --env HOME=/tmp \
    --env TMPDIR=/review-build/tmp \
    --entrypoint /bin/bash \
    "${REVIEW_IMAGE}" \
    -euo pipefail -c '
      UV_PROJECT_ENVIRONMENT=/review-envs/shared \
        uv sync --locked --dev --no-install-workspace --no-cache
      while IFS= read -r project; do
        if [[ "${project}" != root ]]; then
          UV_PROJECT_ENVIRONMENT=/review-envs/shared \
            uv sync --project "projects/${project}" --locked --dev --all-extras \
              --no-install-project --inexact --no-cache
        fi
      done </review-envs/selected-projects
    ' || prepare_status=$?
  sudo rm -rf -- "${REVIEW_BUILD_ROOT}"
  ((prepare_status == 0)) || return "${prepare_status}"
  sudo chown -R root:root "${REVIEW_ENV_ROOT}"
  sudo chmod -R go-w "${REVIEW_ENV_ROOT}"
}

run_test() {
  [[ $# -eq 2 ]] || die "usage: review-falsify {root|typo-cot|typo-robust-training} TEST_FILE"
  local project="$1"
  local host_test_mount python_path pythonpath test_file test_mount test_relative workdir workspace
  workspace="$(require_prepared_workspace)"
  if ! test_file="$(realpath --canonicalize-existing -- "$2" 2>/dev/null)"; then
    die "TEST_FILE must be an existing Python file below ${REVIEW_TEST_ROOT}"
  fi
  [[ -f "${test_file}" && "${test_file}" == "${REVIEW_TEST_ROOT}/"*.py ]] \
    || die "TEST_FILE must be an existing Python file below ${REVIEW_TEST_ROOT}"
  test_relative="${test_file#${REVIEW_TEST_ROOT}/}"

  case "${project}" in
    root)
      python_path="/review-envs/shared/bin/python"
      pythonpath="/workspace"
      host_test_mount="${workspace}/tests/.review-tests"
      test_mount="/workspace/tests/.review-tests"
      workdir="/workspace"
      ;;
    typo-cot | typo-robust-training)
      grep -Fxq -- "${project}" "${REVIEW_ENV_ROOT}/selected-projects" \
        || die "the ${project} environment was not provisioned because that project is unchanged"
      python_path="/review-envs/shared/bin/python"
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
  if [[ ! -f "${REVIEW_ENV_ROOT}/shared/pyvenv.cfg" ]] \
    || { [[ ! -L "${host_python}" ]] && [[ ! -f "${host_python}" ]]; }; then
    die "the ${project} review environment is not installed"
  fi
  # Tests are authored by the runner but executed by the unprivileged sandbox.
  chmod 0644 -- "${test_file}"

  local test_container_name
  test_container_name="review-falsify-test-${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-0}-$$-${RANDOM}"
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
    --env "PYTHONPATH=${pythonpath}" \
    --env PYTHONDONTWRITEBYTECODE=1 \
    --entrypoint "${python_path}" \
    "${REVIEW_IMAGE}" \
    -m pytest "${test_mount}/${test_relative}" -q --disable-warnings --maxfail=1
}

self_test() {
  [[ $# -eq 0 ]] || die "usage: review-falsify --self-test"
  require_prepared_workspace >/dev/null
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
