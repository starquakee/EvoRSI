"""Capture immutable source bytes without following links or moving uploads."""
from __future__ import annotations

import errno
import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path

from .source_gate import GateDenial, SourcePolicy, SourceVerdict, check_source_bytes

MAX_ARTIFACT_BYTES = 8 * 1024 * 1024
MAX_ARTIFACT_FILES = 1024


class CaptureError(ValueError):
    def __init__(self, rule: str, reason: str):
        super().__init__(reason)
        self.rule = rule
        self.reason = reason


@dataclass(frozen=True)
class SourceSnapshot:
    name: str
    is_directory: bool
    files: tuple[tuple[str, bytes], ...]
    verdict: SourceVerdict

    def materialize(self, code_root: Path) -> Path:
        """Write only captured bytes into a fresh, trusted admission directory."""
        if not self.verdict.allowed:
            raise CaptureError('source_gate_denied', 'cannot materialize denied artifact')
        target = code_root / self.name
        if self.is_directory:
            target.mkdir(exist_ok=False)
        for relative, data in self.files:
            path = target / relative if self.is_directory else target
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
            try:
                with os.fdopen(descriptor, 'wb', closefd=False) as stream:
                    stream.write(data)
            finally:
                os.close(descriptor)
        return target


def _read_regular(descriptor: int, limit: int) -> bytes:
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        raise CaptureError('artifact_not_regular', 'source artifact is not a regular file')
    if metadata.st_nlink != 1:
        raise CaptureError('hardlink_denied', 'hardlinked source artifact is unsupported')
    if metadata.st_size > limit:
        raise CaptureError('source_too_large', 'source artifact exceeds byte bound')
    chunks: list[bytes] = []
    remaining = limit + 1
    while remaining:
        chunk = os.read(descriptor, min(65536, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    data = b''.join(chunks)
    if len(data) > limit:
        raise CaptureError('source_too_large', 'source artifact grew past byte bound')
    return data


def capture_source(path: str | os.PathLike[str], policy: SourcePolicy, *, allowed_root: str | os.PathLike[str] | None = None) -> SourceSnapshot:
    """Read once, validate those bytes, and retain exactly what admission writes.

    Ancestors and descendants are opened with O_NOFOLLOW relative to pinned
    directory descriptors. No file-system operations mutate the upload.
    All files contribute to identity, including non-Python assets.
    """
    raw = Path(path)
    if '..' in raw.parts:
        raise CaptureError('path_traversal', 'source path contains traversal')
    absolute = Path(os.path.abspath(raw))
    if allowed_root is not None:
        root = Path(os.path.abspath(allowed_root))
        if not absolute.is_relative_to(root):
            raise CaptureError('path_traversal', 'source path escapes storage')
    if not absolute.name:
        raise CaptureError('artifact_not_regular', 'filesystem root is not source')
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    file_flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
    descriptors: list[int] = []
    files: list[tuple[str, bytes]] = []
    total_bytes = 0
    visited_entries = 0

    def append_file(relative: str, descriptor: int) -> None:
        nonlocal total_bytes
        if len(files) >= MAX_ARTIFACT_FILES:
            raise CaptureError('source_too_large', 'too many source files')
        data = _read_regular(descriptor, MAX_ARTIFACT_BYTES - total_bytes)
        total_bytes += len(data)
        files.append((relative, data))

    def descend(descriptor: int, prefix: str = '', depth: int = 0) -> None:
        nonlocal visited_entries
        if depth > 32:
            raise CaptureError('source_too_large', 'source tree too deep')
        with os.scandir(descriptor) as entries:
            for entry in entries:
                visited_entries += 1
                if visited_entries > MAX_ARTIFACT_FILES * 2:
                    raise CaptureError('source_too_large', 'too many artifact entries')
                name = entry.name
                relative = prefix + name
                try:
                    relative.encode('utf-8')
                except UnicodeEncodeError as exc:
                    raise CaptureError('path_invalid', 'non-UTF8 source filename') from exc
                child = os.open(name, file_flags, dir_fd=descriptor)
                try:
                    metadata = os.fstat(child)
                    if stat.S_ISDIR(metadata.st_mode):
                        descend(child, relative + '/', depth + 1)
                    else:
                        append_file(relative, child)
                finally:
                    os.close(child)

    try:
        current = os.open('/', directory_flags)
        descriptors.append(current)
        for part in absolute.parts[1:-1]:
            current = os.open(part, directory_flags, dir_fd=current)
            descriptors.append(current)
        artifact = os.open(absolute.name, file_flags, dir_fd=current)
        descriptors.append(artifact)
        is_directory = stat.S_ISDIR(os.fstat(artifact).st_mode)
        if is_directory:
            descend(artifact)
        else:
            append_file(absolute.name, artifact)
    except OSError as exc:
        rule = 'symlink_denied' if exc.errno in (errno.ELOOP, errno.ENOTDIR) else 'artifact_unreadable'
        if exc.errno == errno.ENOENT:
            rule = 'artifact_missing'
        raise CaptureError(rule, type(exc).__name__) from exc
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)

    denials: list[GateDenial] = []
    python_files = 0
    tree_hash = hashlib.sha256()
    for relative, data in sorted(files):
        if not is_directory or relative.endswith('.py'):
            python_files += 1
            denials.extend(check_source_bytes(data, policy).denials)
        tree_hash.update(relative.encode('utf-8') + b'\0')
        tree_hash.update(hashlib.sha256(data).hexdigest().encode('ascii') + b'\n')
    if not python_files:
        denials.append(GateDenial('empty_source', 'no Python source in artifact'))
    verdict = SourceVerdict(
        allowed=not denials, policy_version=policy.policy_version,
        policy_sha256=policy.policy_sha256, source_sha256=tree_hash.hexdigest(),
        denials=tuple(denials),
    )
    return SourceSnapshot(absolute.name, is_directory, tuple(files), verdict)
