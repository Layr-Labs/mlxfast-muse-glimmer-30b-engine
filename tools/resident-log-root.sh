#!/usr/bin/env bash
# resident-log-root.sh prepare ROOT -- the ONE fixed parent of every
# measurement's resident log directory.
#
# Each measurement gets a new log directory (resident-up.XXXXXX) INSIDE ROOT.
# A previous measurement whose stop failed can leave its log directory, its
# retained resident logs or a resident's private tmp.* tree (written by entry
# code) in ROOT. Before a new measurement boots any leg, this script:
#   1. refuses a ROOT that is a link, not a directory, not owned by this
#      account, or writable by group or other (it creates ROOT mode 0700);
#   2. removes every entry in ROOT, restoring owner access entry by entry first
#      (entry code can leave mode-000 directories and user-immutable files),
#      and proves each one is gone;
#   3. refuses (exit 2) when any entry survives -- a leftover that cannot be
#      removed means no measurement;
#   4. creates this measurement's log directory in ROOT and prints its path.
# tools/resident-up.sh denies entry reads and writes of the WHOLE ROOT
# (RESIDENT_UP_LOG_ROOT), except the leg's own tmp.* directory, and refuses to
# boot while ROOT holds anything but the current log directory.
set -uo pipefail
refuse() { printf 'resident-log-root.sh: REFUSED (%s): %s\n' "$1" "$2" >&2; exit 2; }
[[ $# -eq 2 && "$1" == "prepare" && -n "$2" ]] || { echo "usage: resident-log-root.sh prepare ROOT" >&2; exit 2; }
ROOT="$2"
if [[ ! -e "${ROOT}" && ! -L "${ROOT}" ]]; then
  mkdir -m 700 -p "${ROOT}" || refuse log-root-unusable "cannot create ${ROOT}"
fi
[[ ! -L "${ROOT}" && -d "${ROOT}" && -O "${ROOT}" ]] \
  || refuse log-root-unusable "${ROOT} is a link, not a directory, or not owned by this account"
# GNU stat first (hosted Ubuntu CI), BSD stat on macOS.
mode="$(stat -c '%a' "${ROOT}" 2>/dev/null || stat -f '%Lp' "${ROOT}" 2>/dev/null)"
[[ "${mode}" =~ ^[0-7]+$ ]] || refuse log-root-unusable "cannot read the mode of ${ROOT}"
(( (8#${mode} & 8#022) == 0 )) || refuse log-root-unusable "${ROOT} is writable by group or other (mode ${mode})"
# Restore owner access to one path, never through a link: macOS clears the
# user flags and sets the mode on the link itself (-h); GNU chmod has no -h and
# Linux has no chflags, so there a link is left as it is (rm removes the link).
UNLOCK='if [ -x /usr/bin/chflags ]; then /usr/bin/chflags -h nouchg,nouappnd "$1"; /bin/chmod -h u+rwx "$1"; elif [ ! -L "$1" ]; then chmod u+rwx "$1"; fi'
while IFS= read -r -d '' entry; do
  /bin/sh -c "${UNLOCK}" sh "${entry}" 2>/dev/null
  if [[ -d "${entry}" && ! -L "${entry}" ]]; then
    find "${entry}" -exec /bin/sh -c "${UNLOCK}" sh {} \; 2>/dev/null
  fi
  rm -rf -- "${entry}" 2>/dev/null
  [[ ! -e "${entry}" && ! -L "${entry}" ]] \
    || refuse measurement-leftover "an earlier measurement's ${entry} cannot be removed. Nothing has been loaded."
done < <(find "${ROOT}" -mindepth 1 -maxdepth 1 -print0 2>/dev/null)
left="$(find "${ROOT}" -mindepth 1 -maxdepth 1 -print 2>&1)" \
  || refuse measurement-leftover "cannot list ${ROOT}: ${left}. Nothing has been loaded."
[[ -z "${left}" ]] || refuse measurement-leftover "${ROOT} still holds: ${left}. Nothing has been loaded."
mktemp -d "${ROOT}/resident-up.XXXXXX" || refuse log-root-unusable "cannot create a log directory in ${ROOT}"
