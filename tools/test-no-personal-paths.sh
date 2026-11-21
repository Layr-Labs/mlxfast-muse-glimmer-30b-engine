#!/usr/bin/env bash
#
# test-no-personal-paths.sh -- the tree names no person's home directory.
#
# This repository's history is published. A path such as /Users/<name>/... or
# /home/<name>/... names a real person and their machine layout. Examples use a
# neutral path (/path/to/...). Only the CI and service account homes below are
# allowed: they name no person.
#
# In a git work tree this scans the committed tree (git grep); outside one (an
# exported archive) it scans the files on disk.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PATTERN='/Users/[A-Za-z]|/home/[a-z]'
# /Users/runner: the GitHub-hosted macOS runner account and the box-runner
# account. /Users/Shared: the macOS shared folder.
ALLOWED='/(Users|home)/(runner|Shared)([^A-Za-z0-9._-]|$)'

cd "${REPO_ROOT}"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  hits="$(git grep -nIE "${PATTERN}" -- . || true)"
else
  hits="$(grep -rnIE "${PATTERN}" . || true)"
fi

bad=""
while IFS= read -r line; do
  [[ -z "${line}" ]] && continue
  # Drop every allowed home, then keep the line only if a home is left.
  rest="$(sed -E "s#${ALLOWED}#\\3#g" <<<"${line}")"
  if grep -qE "${PATTERN}" <<<"${rest}"; then
    bad+="${line}"$'\n'
  fi
done <<<"${hits}"

if [[ -z "${bad}" ]]; then
  echo "PASS no person home path in the tree"
  echo "test-no-personal-paths.sh: all cases passed"
else
  printf 'FAIL person home path: %s\n' "${bad}" | cut -c1-200
  echo "test-no-personal-paths.sh: 1 case(s) failed"
  exit 1
fi
