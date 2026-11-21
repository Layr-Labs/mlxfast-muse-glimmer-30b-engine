#!/usr/bin/env bash
#
# editable-paths-doc.sh -- print the editable path list from benchmark.json,
# and keep the copies of it in the participant docs in step.
#
# benchmark.json `editablePaths` is the authority for what a submission
# uploads. The docs used to restate part of it by hand, and the copy fell
# behind: it named DFlash2Draft.swift after the list had moved to the Muse
# Glimmer target and MuseGlimmerDFlashDrafter.swift. A file edited from a stale
# list does not upload, so the docs now carry a block this script generates.
#
# Usage:
#   tools/editable-paths-doc.sh            print every entry, grouped
#   tools/editable-paths-doc.sh block      print the generated doc block
#   tools/editable-paths-doc.sh --check    exit 1 when a doc block differs
#   tools/editable-paths-doc.sh --write    rewrite the doc blocks in place
#
# The doc block sits between these two lines in README.md and
# docs/participant-contract.md:
#
#   <!-- BEGIN GENERATED: editable model files (tools/editable-paths-doc.sh) -->
#   <!-- END GENERATED: editable model files -->
#
# An entry that fits no group is an error, so a new kind of entry cannot be
# left out of the docs in silence.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
MANIFEST="${EDITABLE_PATHS_MANIFEST:-${REPO_ROOT}/benchmark.json}"
DOCS=("README.md" "docs/participant-contract.md")
BEGIN_MARK="<!-- BEGIN GENERATED: editable model files (tools/editable-paths-doc.sh) -->"
END_MARK="<!-- END GENERATED: editable model files -->"

die() {
  echo "editable-paths-doc.sh: $*" >&2
  exit 1
}

command -v jq >/dev/null 2>&1 || die "jq is required to read ${MANIFEST}"
entries="$(jq -r '.editablePaths[]' "${MANIFEST}")" || die "cannot read editablePaths from ${MANIFEST}"
[[ -n "${entries}" ]] || die "${MANIFEST} lists no editablePaths"

# group_of <path> -> declaration | transform | model | engine | kernels
group_of() {
  case "$1" in
    mtp-head.manifest.json) echo declaration ;;
    Sources/MLXFastTransform|Sources/MLXFastTransform/*) echo transform ;;
    Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/*) echo model ;;
    Vendor/mlx-swift-lm/Libraries/MLXLMCommon/PrismHadamardCheckpoint.swift) echo model ;;
    Vendor/mlx-swift/Source/MLXNN/*) echo model ;;
    Vendor/mlx-swift-lm/Libraries/MLXLMCommon/*|Vendor/mlx-swift-lm/Libraries/MLXLMServer/*) echo engine ;;
    Vendor/mlx-swift/Source/Cmlx/*) echo kernels ;;
    *) return 1 ;;
  esac
}

entries_in() {
  local want="$1" path group
  while IFS= read -r path; do
    group="$(group_of "${path}")" || die "editablePaths entry fits no group: ${path} (add it to group_of)"
    [[ "${group}" == "${want}" ]] && printf '%s\n' "${path}"
  done <<<"${entries}"
  return 0
}

total="$(wc -l <<<"${entries}" | tr -d '[:space:]')"

print_all() {
  local group title count
  for group in declaration transform model engine kernels; do
    case "${group}" in
      declaration) title="The decoder declaration" ;;
      transform) title="The offline transform" ;;
      model) title="The model files" ;;
      engine) title="The batching engine and the cache layer" ;;
      kernels) title="The vendored MLX Metal kernels" ;;
    esac
    count="$(entries_in "${group}" | grep -c . || true)"
    printf '%s (%s)\n' "${title}" "${count}"
    entries_in "${group}" | sed 's/^/  /'
  done
  printf 'Total: %s editablePaths entries\n' "${total}"
}

render_block() {
  local count
  count="$(entries_in model | grep -c . || true)"
  printf '%s\n' "${BEGIN_MARK}"
  printf 'The %s model files on the editable surface, of the %s `benchmark.json`\n' "${count}" "${total}"
  printf '`editablePaths` entries. This list is generated from `benchmark.json`;\n'
  printf 'run `tools/editable-paths-doc.sh --write` after the list changes.\n\n'
  entries_in model | sed 's/^/- `/; s/$/`/'
  printf '%s\n' "${END_MARK}"
}

# The current block in a doc, markers included; empty when it has none.
current_block() {
  awk -v begin="${BEGIN_MARK}" -v end="${END_MARK}" '
    $0 == begin { inside = 1 }
    inside { print }
    $0 == end { inside = 0 }
  ' "$1"
}

case "${1:-}" in
  "")
    print_all
    ;;
  block)
    render_block
    ;;
  --check)
    want="$(render_block)"
    bad=0
    for doc in "${DOCS[@]}"; do
      have="$(current_block "${REPO_ROOT}/${doc}")"
      if [[ -z "${have}" ]]; then
        echo "editable-paths-doc.sh: ${doc} has no generated editable model files block" >&2
        bad=1
      elif [[ "${have}" != "${want}" ]]; then
        echo "editable-paths-doc.sh: ${doc} lists editable model files that differ from benchmark.json; run tools/editable-paths-doc.sh --write" >&2
        diff <(printf '%s\n' "${have}") <(printf '%s\n' "${want}") >&2 || true
        bad=1
      fi
    done
    [[ "${bad}" -eq 0 ]] && echo "editable-paths-doc.sh: the doc blocks match benchmark.json editablePaths"
    exit "${bad}"
    ;;
  --write)
    block_file="$(mktemp)"
    trap 'rm -f "${block_file}"' EXIT
    render_block > "${block_file}"
    for doc in "${DOCS[@]}"; do
      path="${REPO_ROOT}/${doc}"
      grep -qxF "${BEGIN_MARK}" "${path}" || die "${doc} has no '${BEGIN_MARK}' line to write after"
      grep -qxF "${END_MARK}" "${path}" || die "${doc} has no '${END_MARK}' line"
      awk -v begin="${BEGIN_MARK}" -v end="${END_MARK}" -v block="${block_file}" '
        $0 == begin { while ((getline line < block) > 0) print line; close(block); skip = 1; next }
        $0 == end { skip = 0; next }
        !skip { print }
      ' "${path}" > "${path}.tmp"
      mv "${path}.tmp" "${path}"
      echo "editable-paths-doc.sh: wrote ${doc}"
    done
    ;;
  *)
    die "usage: tools/editable-paths-doc.sh [block|--check|--write]"
    ;;
esac
