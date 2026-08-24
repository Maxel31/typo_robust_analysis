"""Fail-closed filesystem primitives for immutable training artifacts."""

from __future__ import annotations

import ctypes
import errno
import os
import stat
from pathlib import Path


def reject_path_symlink_components(path: Path, *, artifact: str) -> None:
    """Reject root and ancestor symlinks before resolution can erase them."""

    supplied = Path(path)
    if ".." in supplied.parts:
        raise ValueError(f"{artifact} path cannot contain parent traversal")
    absolute = supplied if supplied.is_absolute() else Path.cwd() / supplied
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        if part in {"", "."}:
            continue
        cursor /= part
        try:
            mode = cursor.lstat().st_mode
        except FileNotFoundError:
            break
        if stat.S_ISLNK(mode):
            if cursor == absolute:
                raise ValueError(f"{artifact} root cannot be a symlink")
            raise ValueError(f"{artifact} path contains a symlink: {cursor}")


def publish_directory_noreplace(
    staged: Path,
    output: Path,
    *,
    expected_parent_identity: tuple[int, int] | None = None,
) -> None:
    """Atomically publish one sibling directory without replacing an existing path."""

    source = Path(staged)
    target = Path(output)
    if source.parent != target.parent or source.name in {"", ".", ".."}:
        raise ValueError("staged publication must use one sibling directory")
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic no-replace directory publication is unavailable")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    parent_descriptor = os.open(source.parent, directory_flags)
    try:
        opened_parent = os.fstat(parent_descriptor)
        if (
            expected_parent_identity is not None
            and (
                opened_parent.st_dev,
                opened_parent.st_ino,
            )
            != expected_parent_identity
        ):
            raise ValueError("atomic publication parent changed before rename")
        staged_metadata = os.stat(source.name, dir_fd=parent_descriptor, follow_symlinks=False)
        if not stat.S_ISDIR(staged_metadata.st_mode):
            raise ValueError("staged publication source must be one directory")
        result = renameat2(
            parent_descriptor,
            os.fsencode(source.name),
            parent_descriptor,
            os.fsencode(target.name),
            1,  # RENAME_NOREPLACE
        )
        if result != 0:
            error = ctypes.get_errno()
            if error in {errno.EEXIST, errno.ENOTEMPTY}:
                raise FileExistsError(f"output appeared during atomic publication: {target}")
            raise OSError(error, os.strerror(error), target)
        published_metadata = os.stat(
            target.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        visible_metadata = target.lstat()
        if (published_metadata.st_dev, published_metadata.st_ino) != (
            visible_metadata.st_dev,
            visible_metadata.st_ino,
        ):
            raise RuntimeError("atomic publication parent changed during rename")
    finally:
        os.close(parent_descriptor)


__all__ = ["publish_directory_noreplace", "reject_path_symlink_components"]
