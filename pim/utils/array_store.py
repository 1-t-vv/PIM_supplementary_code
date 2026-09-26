from __future__ import annotations

import hashlib
import json
import os
import re
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


ARRAY_STORE_FORMAT = "mesh-array-store"
ARRAY_STORE_VERSION = 1
MANIFEST_NAME = "manifest.json"
_SAFE_KEY = re.compile(r"^[A-Za-z0-9_.-]+$")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_file(path: os.PathLike[str] | str, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _validate_key(key: str) -> str:
    key = str(key)
    if not _SAFE_KEY.fullmatch(key) or key in {".", ".."}:
        raise ValueError(f"Unsafe array-store key: {key!r}")
    return key


def _dataset_fingerprint(array_records: Iterable[Mapping[str, Any]]) -> str:
    identity = [
        {
            "key": record["key"],
            "shape": record["shape"],
            "dtype": record["dtype"],
            "sha256": record["sha256"],
        }
        for record in sorted(array_records, key=lambda item: str(item["key"]))
    ]
    return hashlib.sha256(_canonical_json(identity)).hexdigest()


def _read_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path / MANIFEST_NAME
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except FileNotFoundError as exc:
        raise ValueError(f"Array store is incomplete: missing {manifest_path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid array-store manifest {manifest_path}: {exc}") from exc
    if manifest.get("format") != ARRAY_STORE_FORMAT:
        raise ValueError(f"Unsupported array-store format in {manifest_path}")
    if manifest.get("version") != ARRAY_STORE_VERSION:
        raise ValueError(
            f"Unsupported array-store version {manifest.get('version')} in {manifest_path}"
        )
    if manifest.get("state") != "complete":
        raise ValueError(f"Array store is not complete: {manifest_path}")
    # Stores written before cache_fingerprint was introduced use their content
    # fingerprint for cache identity.
    manifest.setdefault("cache_fingerprint", manifest.get("dataset_fingerprint"))
    return manifest


def open_array_store(
    path: os.PathLike[str] | str,
    *,
    verify_checksums: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    path = Path(path)
    if not path.is_dir():
        raise ValueError(f"Array store must be a directory: {path}")
    manifest = _read_manifest(path)
    records = manifest.get("arrays")
    if not isinstance(records, list) or not records:
        raise ValueError(f"Array-store manifest has no arrays: {path / MANIFEST_NAME}")

    arrays: dict[str, np.ndarray] = {}
    normalized_records: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"Invalid array record in {path / MANIFEST_NAME}")
        key = _validate_key(record.get("key", ""))
        if key in arrays:
            raise ValueError(f"Duplicate array-store key {key!r}")
        filename = record.get("file")
        if filename != f"{key}.npy":
            raise ValueError(f"Unsafe or inconsistent file for field {key!r}: {filename!r}")
        array_path = path / filename
        try:
            stat = array_path.stat()
        except FileNotFoundError as exc:
            raise ValueError(f"Array store is missing {array_path}") from exc
        if stat.st_size != int(record.get("file_size", -1)):
            raise ValueError(
                f"Array-store file size mismatch for {key!r}: "
                f"expected {record.get('file_size')}, got {stat.st_size}"
            )
        if verify_checksums and _sha256_file(array_path) != record.get("sha256"):
            raise ValueError(f"Array-store checksum mismatch for {key!r}")
        try:
            array = np.load(array_path, mmap_mode="r", allow_pickle=False)
        except Exception as exc:
            raise ValueError(f"Cannot open array-store field {key!r}: {exc}") from exc
        expected_shape = tuple(int(v) for v in record.get("shape", []))
        expected_dtype = np.dtype(record.get("dtype"))
        if array.shape != expected_shape or array.dtype != expected_dtype:
            raise ValueError(
                f"Array-store metadata mismatch for {key!r}: "
                f"manifest={expected_shape}/{expected_dtype}, file={array.shape}/{array.dtype}"
            )
        if array.nbytes != int(record.get("data_nbytes", -1)):
            raise ValueError(f"Array-store byte-count mismatch for {key!r}")
        arrays[key] = array
        normalized_records.append(record)

    expected_fingerprint = _dataset_fingerprint(normalized_records)
    if manifest.get("dataset_fingerprint") != expected_fingerprint:
        raise ValueError(f"Array-store fingerprint metadata is inconsistent: {path}")
    cache_fingerprint = manifest.get("cache_fingerprint")
    if not isinstance(cache_fingerprint, str) or len(cache_fingerprint) < 16:
        raise ValueError(f"Array-store cache fingerprint is invalid: {path}")
    return arrays, manifest


def legacy_npz_fingerprint(path: os.PathLike[str] | str) -> str:
    """Fingerprint NPZ content from its ZIP directory without decompressing it."""
    path = Path(path)
    records = []
    try:
        with zipfile.ZipFile(path, "r") as archive:
            for info in sorted(archive.infolist(), key=lambda item: item.filename):
                records.append(
                    {
                        "file": info.filename,
                        "size": int(info.file_size),
                        "crc32": int(info.CRC),
                    }
                )
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError(f"Invalid legacy NPZ {path}: {exc}") from exc
    return hashlib.sha256(_canonical_json(records)).hexdigest()


def resolve_dataset_path(path: os.PathLike[str] | str) -> Path:
    """Resolve a configured store path with bidirectional NPZ compatibility."""
    path = Path(path)
    if path.exists():
        # Prefer an explicitly configured path. This avoids silently selecting
        # a stale sidecar when both formats exist.
        return path
    if path.suffix == ".store":
        legacy = path.with_suffix(".npz")
        if legacy.exists():
            return legacy
    elif path.suffix == ".npz":
        store = path.with_suffix(".store")
        if store.exists():
            return store
    raise FileNotFoundError(f"Dataset not found: {path}")


def load_dataset_arrays(
    path: os.PathLike[str] | str,
    *,
    verify_checksums: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, Any], Path]:
    resolved = resolve_dataset_path(path)
    if resolved.is_dir():
        arrays, manifest = open_array_store(
            resolved, verify_checksums=verify_checksums
        )
        return arrays, manifest, resolved

    try:
        with np.load(resolved, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
    except Exception as exc:
        raise ValueError(f"Cannot load legacy dataset NPZ {resolved}: {exc}") from exc
    manifest = {
        "format": "legacy-npz",
        "version": 1,
        "state": "complete",
        "dataset_fingerprint": legacy_npz_fingerprint(resolved),
    }
    manifest["cache_fingerprint"] = manifest["dataset_fingerprint"]
    return arrays, manifest, resolved

