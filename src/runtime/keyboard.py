"""Nonblocking terminal keys with guaranteed terminal mode restoration."""

import os
import select
import sys
from contextlib import contextmanager


@contextmanager
def terminal_keys():
    if not sys.stdin.isatty() or sys.platform == "win32":
        yield lambda: ""
        return
    import termios
    import tty

    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        def read():
            return os.read(fd, 1).decode("utf-8", errors="ignore") if select.select([fd], [], [], 0)[0] else ""
        yield read
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous)
