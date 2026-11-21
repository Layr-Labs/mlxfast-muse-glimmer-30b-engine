"""The fleet GPU lock: one opener for every trusted caller.

Three trusted scripts use the box GPU lock (default /tmp/mtplx-gpu-exclusive.lock):

  tools/calibrate-box.sh                      hold   (operator account)
  tools/muse-glimmer-30b-measure-and-score.sh hold   (scoring account)
  tools/resident-up.sh                        held   (before each leg boots)

The lock is an flock on an EMPTY, READ-ONLY file. flock needs no write access,
so every account opens the lock O_RDONLY: a lock that one account created
works for the other account on a fresh box, whatever the umask. The file holds
no data, so it cannot carry data from one leg to the next:

  * a new lock is created mode 0444;
  * a lock that is not a regular file, is a symbolic link, has more than one
    link, holds any byte, or is writable by group or other is REFUSED;
  * a lock owned by THIS account with the owner write bit (an old 0644 lock)
    is set to 0444 by `hold`, which runs before any leg; `held` refuses it,
    because by then only entry code could have changed it. The owner write bit
    of a lock that another account owns does not let this account (or entry
    code running as it) write, so it is accepted.

The Seatbelt profile (tools/seatbelt-profile.py) also denies every read and
write of the lock path to entry code.

  gpu-lock.py hold LOCK TIMEOUT_S WHO ACTION PROGRAM [ARG...]
      wait up to TIMEOUT_S (refusal: "WHO: ... refusing to ACTION beside
      another GPU owner") for an exclusive flock, then execv PROGRAM with the
      lock descriptor inherited, so the lock lives as long as PROGRAM.
      exit 1: timeout; exit 2: unsafe lock (nothing runs).
  gpu-lock.py held LOCK
      exit 0: someone holds it; 1: nobody holds it; 2: missing or unsafe.
"""

import fcntl
import os
import signal
import stat
import sys

NEW_LOCK_MODE = 0o444


def unsafe_reason(st, repairable):
    if not stat.S_ISREG(st.st_mode):
        return "is not a regular file"
    if st.st_nlink != 1:
        return "has %d links" % st.st_nlink
    if st.st_size != 0:
        return "holds %d byte(s); the lock must hold no data" % st.st_size
    if st.st_mode & 0o022:
        return "is writable by group or other (mode %04o)" % (st.st_mode & 0o7777)
    if st.st_mode & 0o200 and st.st_uid == os.geteuid() and not repairable:
        return "is writable by this account (mode %04o)" % (st.st_mode & 0o7777)
    return ""


def open_lock(path, create, repairable):
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | (os.O_CREAT if create else 0)
    old_umask = os.umask(0o022)
    try:
        fd = os.open(path, flags, NEW_LOCK_MODE)
    finally:
        os.umask(old_umask)
    st = os.fstat(fd)
    reason = unsafe_reason(st, repairable)
    if reason:
        os.close(fd)
        raise ValueError(reason)
    if st.st_mode & 0o200 and st.st_uid == os.geteuid():
        os.fchmod(fd, NEW_LOCK_MODE)
        reason = unsafe_reason(os.fstat(fd), False)
        if reason:
            os.close(fd)
            raise ValueError(reason)
    return fd


def refuse(who, path, message, code):
    sys.stderr.write("%s: the GPU lock %s %s. Nothing has been loaded.\n" % (who, path, message))
    return code


def hold(argv):
    if len(argv) < 5:
        sys.stderr.write("usage: gpu-lock.py hold LOCK TIMEOUT_S WHO ACTION PROGRAM [ARG...]\n")
        return 2
    path, timeout_s, who, action, program = argv[0], int(argv[1]), argv[2], argv[3], argv[4:]
    try:
        fd = open_lock(path, create=True, repairable=True)
    except (OSError, ValueError) as error:
        return refuse(who, path, str(error), 2)

    def on_timeout(*_):
        refuse(who, path, "has been held by another owner for %ds; refusing to %s beside "
               "another GPU owner" % (timeout_s, action), 1)
        sys.exit(1)

    signal.signal(signal.SIGALRM, on_timeout)
    signal.alarm(timeout_s)
    fcntl.flock(fd, fcntl.LOCK_EX)
    signal.alarm(0)
    os.set_inheritable(fd, True)
    os.execv(program[0], program)
    return 2


def held(argv):
    if len(argv) != 1:
        sys.stderr.write("usage: gpu-lock.py held LOCK\n")
        return 2
    try:
        fd = open_lock(argv[0], create=False, repairable=False)
    except (OSError, ValueError) as error:
        sys.stderr.write("the GPU lock %s %s\n" % (argv[0], error))
        return 2
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return 0
    fcntl.flock(fd, fcntl.LOCK_UN)
    return 1


def main():
    verbs = {"hold": hold, "held": held}
    if len(sys.argv) < 2 or sys.argv[1] not in verbs:
        sys.stderr.write("usage: gpu-lock.py hold|held ...\n")
        return 2
    return verbs[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    sys.exit(main())
