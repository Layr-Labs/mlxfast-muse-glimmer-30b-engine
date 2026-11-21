#!/usr/bin/env bash
# remove-private-tree.sh PATH... -- remove trees that confined entry code could
# write, and PROVE each one is gone. Entry code can leave entries the owner
# cannot list or remove as they are (mode 000, the user-immutable flag), so
# owner access is restored entry by entry, top down, before the removal.
# A path that does not exist is fine. A path not owned by this account is not
# touched. Exit 1 when any path survives, so the caller fails closed.
set -uo pipefail
status=0
for path in "$@"; do
  [[ -e "${path}" || -L "${path}" ]] || continue
  if [[ ! -O "${path}" ]]; then
    printf 'remove-private-tree.sh: %s is not owned by this account; not removed\n' "${path}" >&2
    status=1; continue
  fi
  if [[ -d "${path}" && ! -L "${path}" ]]; then
    find "${path}" -exec /bin/sh -c '/usr/bin/chflags -h nouchg,nouappnd "$1"; /bin/chmod -h u+rwx "$1"' sh {} \; 2>/dev/null
  else
    /usr/bin/chflags -h nouchg,nouappnd "${path}" 2>/dev/null
  fi
  rm -rf -- "${path}" 2>/dev/null
  if [[ -e "${path}" || -L "${path}" ]]; then
    printf 'remove-private-tree.sh: FAILED to remove %s\n' "${path}" >&2
    status=1
  fi
done
exit "${status}"
