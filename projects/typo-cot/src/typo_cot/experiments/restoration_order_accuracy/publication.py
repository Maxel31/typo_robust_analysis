"""Atomic no-replace publication for completed result directories."""

from __future__ import annotations

import ctypes
import errno
import os
from pathlib import Path

_AT_FDCWD = -100
_RENAME_NOREPLACE = 1


def rename_directory_no_replace(source: Path, destination: Path) -> None:
    """Atomically rename ``source`` while refusing to replace ``destination``."""
    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as exc:
        raise RuntimeError(
            "atomic no-replace publication requires Linux renameat2"
        ) from exc

    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return

    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise FileExistsError(
            error,
            f"output directory already exists: {destination}",
            str(destination),
        )
    raise OSError(error, os.strerror(error), str(destination))
