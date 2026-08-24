"""Fail-closed filesystem primitives for immutable training artifacts."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import secrets
import stat
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DirectoryNodeAttestation:
    relative_path: str
    kind: str
    device: int
    inode: int
    mode: int
    links: int
    size: int
    mtime_ns: int
    ctime_ns: int
    sha256: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "kind": self.kind,
            "device": self.device,
            "inode": self.inode,
            "mode": self.mode,
            "links": self.links,
            "size": self.size,
            "mtime_ns": self.mtime_ns,
            "ctime_ns": self.ctime_ns,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class DirectoryTreeAttestation:
    nodes: tuple[DirectoryNodeAttestation, ...]
    sha256: str


def _metadata_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _descriptor_sha256(descriptor: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while True:
        chunk = os.pread(descriptor, 1024 * 1024, offset)
        if not chunk:
            return digest.hexdigest()
        digest.update(chunk)
        offset += len(chunk)


def _node_attestation(
    relative_path: str,
    kind: str,
    metadata: os.stat_result,
    *,
    sha256: str | None,
) -> DirectoryNodeAttestation:
    return DirectoryNodeAttestation(
        relative_path=relative_path,
        kind=kind,
        device=metadata.st_dev,
        inode=metadata.st_ino,
        mode=metadata.st_mode,
        links=metadata.st_nlink,
        size=metadata.st_size,
        mtime_ns=metadata.st_mtime_ns,
        ctime_ns=metadata.st_ctime_ns,
        sha256=sha256,
    )


def _attest_directory_descriptor(descriptor: int) -> DirectoryTreeAttestation:
    root_metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise ValueError("tree attestation root must be one directory")
    root_device = root_metadata.st_dev
    nodes: list[DirectoryNodeAttestation] = []
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)

    def walk(directory_descriptor: int, relative_path: str) -> None:
        before = os.fstat(directory_descriptor)
        if not stat.S_ISDIR(before.st_mode) or before.st_dev != root_device:
            raise ValueError("attested tree contains a substituted or mounted directory")
        names = tuple(sorted(os.listdir(directory_descriptor)))
        nodes.append(_node_attestation(relative_path, "directory", before, sha256=None))
        for name in names:
            if not isinstance(name, str) or name in {"", ".", ".."} or "/" in name:
                raise ValueError("attested tree contains an invalid node name")
            child_path = name if relative_path == "." else f"{relative_path}/{name}"
            try:
                visible = os.stat(
                    name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise ValueError("attested tree changed during inventory") from exc
            if visible.st_dev != root_device:
                raise ValueError("attested tree crosses a filesystem boundary")
            if stat.S_ISDIR(visible.st_mode):
                try:
                    child_descriptor = os.open(name, directory_flags, dir_fd=directory_descriptor)
                except OSError as exc:
                    raise ValueError("attested tree directory changed during inventory") from exc
                try:
                    opened = os.fstat(child_descriptor)
                    if _metadata_identity(opened) != _metadata_identity(visible):
                        raise ValueError("attested tree directory was substituted")
                    walk(child_descriptor, child_path)
                    final = os.fstat(child_descriptor)
                    final_visible = os.stat(
                        name,
                        dir_fd=directory_descriptor,
                        follow_symlinks=False,
                    )
                    if _metadata_identity(final) != _metadata_identity(
                        opened
                    ) or _metadata_identity(final_visible) != _metadata_identity(opened):
                        raise ValueError("attested tree directory changed during inventory")
                finally:
                    os.close(child_descriptor)
            elif stat.S_ISREG(visible.st_mode):
                if visible.st_nlink != 1:
                    raise ValueError("attested tree contains a hard-linked regular file")
                try:
                    child_descriptor = os.open(name, file_flags, dir_fd=directory_descriptor)
                except OSError as exc:
                    raise ValueError("attested tree file changed during inventory") from exc
                try:
                    opened = os.fstat(child_descriptor)
                    if (
                        not stat.S_ISREG(opened.st_mode)
                        or opened.st_nlink != 1
                        or _metadata_identity(opened) != _metadata_identity(visible)
                    ):
                        raise ValueError("attested tree file was substituted")
                    digest = _descriptor_sha256(child_descriptor)
                    final = os.fstat(child_descriptor)
                    final_visible = os.stat(
                        name,
                        dir_fd=directory_descriptor,
                        follow_symlinks=False,
                    )
                    if _metadata_identity(final) != _metadata_identity(
                        opened
                    ) or _metadata_identity(final_visible) != _metadata_identity(opened):
                        raise ValueError("attested tree file changed during inventory")
                    nodes.append(
                        _node_attestation(
                            child_path,
                            "regular",
                            opened,
                            sha256=digest,
                        )
                    )
                finally:
                    os.close(child_descriptor)
            else:
                raise ValueError("attested tree contains a symlink or special node")
        after = os.fstat(directory_descriptor)
        if (
            _metadata_identity(after) != _metadata_identity(before)
            or tuple(sorted(os.listdir(directory_descriptor))) != names
        ):
            raise ValueError("attested tree directory changed during inventory")

    walk(descriptor, ".")
    ordered = tuple(sorted(nodes, key=lambda node: node.relative_path))
    digest = hashlib.sha256(
        json.dumps(
            [node.as_dict() for node in ordered],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return DirectoryTreeAttestation(nodes=ordered, sha256=digest)


def attest_directory_tree_at(
    parent_descriptor: int,
    name: str,
    *,
    expected_root_identity: tuple[int, int] | None = None,
) -> DirectoryTreeAttestation:
    """Attest every directory and file below one child of a pinned parent FD."""

    if not isinstance(name, str) or name in {"", ".", ".."} or Path(name).name != name:
        raise ValueError("tree attestation name must be one path component")
    parent_metadata = os.fstat(parent_descriptor)
    if not stat.S_ISDIR(parent_metadata.st_mode):
        raise ValueError("tree attestation parent must be one directory")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    except OSError as exc:
        raise ValueError("tree attestation root must be one directory") from exc
    try:
        opened = os.fstat(descriptor)
        visible = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        identity = (opened.st_dev, opened.st_ino)
        if identity != (visible.st_dev, visible.st_ino) or (
            expected_root_identity is not None and identity != expected_root_identity
        ):
            raise ValueError("tree attestation root was substituted")
        result = _attest_directory_descriptor(descriptor)
        final = os.fstat(descriptor)
        final_visible = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        if _metadata_identity(final) != _metadata_identity(opened) or _metadata_identity(
            final_visible
        ) != _metadata_identity(opened):
            raise ValueError("tree attestation root changed during inventory")
        return result
    finally:
        os.close(descriptor)


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


def _renameat2_noreplace(
    parent_descriptor: int,
    source_name: str,
    target_name: str,
) -> int:
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
    result = renameat2(
        parent_descriptor,
        os.fsencode(source_name),
        parent_descriptor,
        os.fsencode(target_name),
        1,  # RENAME_NOREPLACE
    )
    return 0 if result == 0 else ctypes.get_errno()


def _quarantine_name_at(parent_descriptor: int, name: str) -> str:
    for _ in range(100):
        quarantine_name = f".{name}.invalid-{secrets.token_hex(8)}"
        error = _renameat2_noreplace(parent_descriptor, name, quarantine_name)
        if error == 0:
            return quarantine_name
        if error not in {errno.EEXIST, errno.ENOTEMPTY}:
            raise OSError(error, os.strerror(error), name)
    raise FileExistsError("could not allocate a unique publication quarantine")


def publish_directory_noreplace(
    staged: Path,
    output: Path,
    *,
    expected_parent_identity: tuple[int, int] | None = None,
    expected_staged_identity: tuple[int, int] | None = None,
    expected_tree_attestation: DirectoryTreeAttestation | None = None,
) -> None:
    """Reattest a closed-world sibling tree and immediately publish it no-replace."""

    source = Path(staged)
    target = Path(output)
    if source.parent != target.parent or source.name in {"", ".", ".."}:
        raise ValueError("staged publication must use one sibling directory")
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    parent_descriptor = os.open(source.parent, directory_flags)
    staged_descriptor: int | None = None
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
        if (
            expected_staged_identity is not None
            and (
                staged_metadata.st_dev,
                staged_metadata.st_ino,
            )
            != expected_staged_identity
        ):
            raise ValueError("staged publication source changed before rename")
        try:
            staged_descriptor = os.open(source.name, directory_flags, dir_fd=parent_descriptor)
        except OSError as exc:
            raise ValueError("staged publication source changed before rename") from exc
        opened_staged = os.fstat(staged_descriptor)
        if (opened_staged.st_dev, opened_staged.st_ino) != (
            staged_metadata.st_dev,
            staged_metadata.st_ino,
        ):
            raise ValueError("staged publication source changed before rename")
        if expected_tree_attestation is not None:
            observed_tree = _attest_directory_descriptor(staged_descriptor)
            if observed_tree != expected_tree_attestation:
                raise ValueError("staged publication tree changed before rename")
        final_staged = os.stat(
            source.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        pinned_staged = os.fstat(staged_descriptor)
        if (
            not stat.S_ISDIR(final_staged.st_mode)
            or (final_staged.st_dev, final_staged.st_ino)
            != (pinned_staged.st_dev, pinned_staged.st_ino)
            or (
                expected_staged_identity is not None
                and (final_staged.st_dev, final_staged.st_ino) != expected_staged_identity
            )
        ):
            raise ValueError("staged publication source changed before rename")
        error = _renameat2_noreplace(parent_descriptor, source.name, target.name)
        if error != 0:
            if error in {errno.EEXIST, errno.ENOTEMPTY}:
                raise FileExistsError(f"output appeared during atomic publication: {target}")
            raise OSError(error, os.strerror(error), target)
        published_metadata = os.stat(
            target.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        try:
            visible_metadata = target.lstat()
        except OSError as exc:
            quarantine_name = _quarantine_name_at(parent_descriptor, target.name)
            raise RuntimeError(
                "atomic publication path changed during rename and was quarantined as "
                f"{quarantine_name}"
            ) from exc
        published_identity = (published_metadata.st_dev, published_metadata.st_ino)
        if published_identity != (visible_metadata.st_dev, visible_metadata.st_ino) or (
            expected_staged_identity is not None and published_identity != expected_staged_identity
        ):
            quarantine_name = _quarantine_name_at(parent_descriptor, target.name)
            raise RuntimeError(
                f"atomic publication changed during rename and was quarantined as {quarantine_name}"
            )
    finally:
        if staged_descriptor is not None:
            os.close(staged_descriptor)
        os.close(parent_descriptor)


def quarantine_directory_at(
    parent_descriptor: int,
    name: str,
    *,
    expected_parent_identity: tuple[int, int],
    expected_directory_identity: tuple[int, int],
) -> str:
    """Atomically hide one exact published directory before caller-controlled removal."""

    if not isinstance(name, str) or name in {"", ".", ".."} or Path(name).name != name:
        raise ValueError("quarantine name must be one path component")
    parent = os.fstat(parent_descriptor)
    if (
        not stat.S_ISDIR(parent.st_mode)
        or (parent.st_dev, parent.st_ino) != expected_parent_identity
    ):
        raise ValueError("quarantine parent changed")
    published = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    if (
        not stat.S_ISDIR(published.st_mode)
        or (
            published.st_dev,
            published.st_ino,
        )
        != expected_directory_identity
    ):
        raise ValueError("published directory changed before quarantine")
    return _quarantine_name_at(parent_descriptor, name)


__all__ = [
    "DirectoryNodeAttestation",
    "DirectoryTreeAttestation",
    "attest_directory_tree_at",
    "publish_directory_noreplace",
    "quarantine_directory_at",
    "reject_path_symlink_components",
]
