#!/usr/bin/env bash
set -euo pipefail

echo "[1/4] Running tests..."
test_file="$(find . \
  \( \
    -path './.git' \
    -o -path './.venv' \
    -o -path './.docs_site' \
    -o -path './site' \
    -o -path './.pytest_cache' \
    -o -path './.mypy_cache' \
    -o -path './.ruff_cache' \
    -o -path './dist' \
    -o -path './build' \
    -o -path './node_modules' \
    -o -name '__pycache__' \
    -o -name '*.egg-info' \
  \) -prune \
  -o \( -name 'test_*.py' -o -name '*_test.py' \) \
  -print -quit)"

if [[ -n "${test_file}" ]]; then
  if [[ -n "${OPEN_WAM_PUBLIC_RELEASE_TESTS:-}" ]]; then
    # shellcheck disable=SC2206
    release_tests=(${OPEN_WAM_PUBLIC_RELEASE_TESTS})
  else
    release_tests=(
      tests/test_config_loader.py
      tests/test_static_config_schema.py
      tests/test_mot_runtime_routing.py
      tests/test_mot_modules.py::test_mot_action_then_video_action_only_rollout_skips_predicted_video
      tests/test_mot_modules.py::test_mot_decoupled_action_only_rollout_skips_split_cache_video_denoise
    )
  fi

  existing_release_tests=()
  for release_test in "${release_tests[@]}"; do
    release_path="${release_test%%::*}"
    if [[ -e "${release_path}" ]]; then
      existing_release_tests+=("${release_test}")
    else
      echo "[WARN] Skipping missing release test target: ${release_test}"
    fi
  done

  if [[ "${#existing_release_tests[@]}" -gt 0 ]]; then
    python3 -m pytest "${existing_release_tests[@]}"
  else
    echo "[WARN] No release test targets found. Skipping pytest."
  fi
else
  echo "[WARN] No pytest test files found. Skipping pytest."
fi

find_private_files() {
  find . \
    \( \
      -path './.git' \
      -o -path './.venv' \
      -o -path './.docs_site' \
      -o -path './site' \
      -o -path './.pytest_cache' \
      -o -path './.mypy_cache' \
      -o -path './.ruff_cache' \
      -o -path './dist' \
      -o -path './build' \
      -o -path './node_modules' \
      -o -name '__pycache__' \
      -o -name '*.egg-info' \
    \) -prune \
    -o \( \
      -name '.env' \
      -o -name '.env.*' \
      -o -name '*.pem' \
      -o -name '*.key' \
      -o -name '*.p12' \
      -o -name 'credentials.json' \
      -o -name '*secret*' \
    \) \
    -print
}

echo "[2/4] Checking for private files..."
private_files="$(find_private_files)"

if [[ -n "${private_files}" ]]; then
  echo "[ERROR] Potentially private files found:"
  printf '%s\n' "${private_files}"
  exit 1
fi

echo "[3/4] Checking for suspicious text..."
set +e
grep -RniE \
  '(api[_-]?key|access[_-]?token|client[_-]?secret|private[_-]?key|password|confidential)' \
  . \
  --exclude-dir=.git \
  --exclude-dir=.venv \
  --exclude-dir=.docs_site \
  --exclude-dir=site \
  --exclude-dir=.pytest_cache \
  --exclude-dir=.mypy_cache \
  --exclude-dir=.ruff_cache \
  --exclude-dir=dist \
  --exclude-dir=build \
  --exclude-dir=node_modules \
  --exclude-dir=__pycache__ \
  --exclude-dir='*.egg-info' \
  --exclude='check_public_release.sh'
grep_status="$?"
set -e

if [[ "${grep_status}" -eq 0 ]]; then
  echo
  echo "[ERROR] Review suspicious references before publication."
  exit 1
elif [[ "${grep_status}" -gt 1 ]]; then
  echo "[ERROR] Suspicious-text scan failed with grep exit code ${grep_status}."
  exit 1
fi

echo "[4/4] Running secret scanner..."
if command -v gitleaks >/dev/null 2>&1; then
  gitleaks detect --source . --no-git --redact
else
  echo "[WARN] gitleaks is not installed. Skipping the gitleaks scan."
fi

echo "[OK] Public release checks passed."
