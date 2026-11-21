#!/usr/bin/env bash
# Restore the protected default of every optional editable path that the
# candidate commit leaves out, before the hosted checks read the work tree.
#
# This is the rule of overlay-editable-paths.sh (the optional-path branch): an
# ABSENT optional editable path means "keep the trusted copy", never "refuse".
# An honest serial entry may leave out mtp-head.manifest.json; without this
# step the manifest lint and the whole-surface static review read a work tree
# without it and refuse the entry before it is scored.
#
# The optional list and the default copy come from the protected base commit
# (BASE_SHA), never from the candidate. A path that IS present is left as the
# candidate wrote it, so a broken file still reaches the checks and fails them.
set -euo pipefail

: "${BASE_SHA:?BASE_SHA is required (the protected base commit)}"
CONTRACT_PATH="${CONTRACT_PATH:-benchmark.json}"

contract="$(git show "${BASE_SHA}:${CONTRACT_PATH}")"
optional="$(jq -r '(.optionalEditablePaths // [])[]' <<< "${contract}")"

while IFS= read -r path; do
  [[ -n "${path}" ]] || continue
  case "/${path}/" in
    //*|*"/../"*|*"/./"*)
      echo "::error::invalid optional editable path '${path}' in ${BASE_SHA}:${CONTRACT_PATH}" >&2
      exit 1
      ;;
  esac
  [[ ! -e "${path}" && ! -L "${path}" ]] || continue
  git checkout "${BASE_SHA}" -- "${path}"
  echo "benchmark: optional editable path ${path} absent from the candidate; restored the protected default from ${BASE_SHA}"
done <<< "${optional}"
