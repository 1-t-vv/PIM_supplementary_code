from __future__ import annotations

import json
import os
import zlib
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np


KNN_CACHE_FORMAT = "mesh-knn-cache"
KNN_CACHE_VERSION = 1
KNN_ALGORITHM = "exact-euclidean-fp32-self-masked-v1"
FINAL_MANIFEST = "manifest.json"


def _crc32(array: np.ndarray) -> int:
    contiguous = np.ascontiguousarray(array)
    if contiguous.nbytes == 0:
        return 0
    return zlib.crc32(contiguous.tobytes(order="C")) & 0xFFFFFFFF


class KnnCacheStore:
    """Read and validate a packaged aggregate KNN cache."""

    def __init__(
        self,
        cache_dir: os.PathLike[str] | str,
        *,
        dataset_fingerprint: str,
        obj_names: Sequence[str],
        sample_count: int,
        vertex_counts: Mapping[str, int],
        k: int,
        index_dtype: np.dtype,
    ) -> None:
        if sample_count <= 0:
            raise ValueError(f"sample_count must be positive, got {sample_count}")
        if k < 0:
            raise ValueError(f"k must be non-negative, got {k}")
        self.cache_dir = Path(cache_dir)
        self.dataset_fingerprint = str(dataset_fingerprint)
        if len(self.dataset_fingerprint) < 16:
            raise ValueError("dataset_fingerprint is unexpectedly short")
        self.obj_names = [str(name) for name in obj_names]
        if not self.obj_names or len(set(self.obj_names)) != len(self.obj_names):
            raise ValueError("obj_names must be non-empty and unique")
        self.sample_count = int(sample_count)
        self.vertex_counts = {
            name: int(vertex_counts[name]) for name in self.obj_names
        }
        if any(vertices <= 0 for vertices in self.vertex_counts.values()):
            raise ValueError("All objects must have at least one vertex")
        self.k = int(k)
        self.effective_k = {
            name: min(self.k, self.vertex_counts[name] - 1)
            for name in self.obj_names
        }
        self.index_dtype = np.dtype(index_dtype)
        if self.index_dtype.kind != "u":
            raise ValueError(f"KNN index dtype must be unsigned, got {self.index_dtype}")

        identity = f"v{KNN_CACHE_VERSION}-{self.dataset_fingerprint}-k{self.k}"
        self.path = self.cache_dir / identity
        self.final_manifest_path = self.path / FINAL_MANIFEST
        self.completed_path = self.path / "completed.npy"
        self.checksums_path = self.path / "row_crc32.npy"
        self._indices: Optional[dict[str, np.ndarray]] = None
        self._completed: Optional[np.ndarray] = None
        self._checksums: Optional[np.ndarray] = None
        self._opened_state: Optional[str] = None

    def _index_path(self, name: str) -> Path:
        return self.path / f"idx_{name}.npy"

    def _metadata(self, state: str) -> dict[str, Any]:
        return {
            "format": KNN_CACHE_FORMAT,
            "version": KNN_CACHE_VERSION,
            "state": state,
            "algorithm": KNN_ALGORITHM,
            "dataset_fingerprint": self.dataset_fingerprint,
            "sample_count": self.sample_count,
            "obj_names": self.obj_names,
            "vertex_counts": self.vertex_counts,
            "requested_k": self.k,
            "effective_k": self.effective_k,
            "index_dtype": self.index_dtype.str,
            "checksum": "crc32-per-sample-object",
            "arrays": {
                name: {
                    "file": self._index_path(name).name,
                    "shape": [
                        self.sample_count,
                        self.vertex_counts[name],
                        self.effective_k[name],
                    ],
                }
                for name in self.obj_names
            },
        }

    def _validate_manifest(self, manifest: Mapping[str, Any], state: str) -> None:
        expected = self._metadata(state)
        for key in (
            "format",
            "version",
            "state",
            "algorithm",
            "dataset_fingerprint",
            "sample_count",
            "obj_names",
            "vertex_counts",
            "requested_k",
            "effective_k",
            "index_dtype",
            "checksum",
            "arrays",
        ):
            if manifest.get(key) != expected[key]:
                raise ValueError(
                    f"KNN cache metadata mismatch for {key!r} in {self.path}: "
                    f"expected {expected[key]!r}, got {manifest.get(key)!r}"
                )

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                value = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid KNN cache manifest {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"KNN cache manifest is not an object: {path}")
        return value

    def _open_arrays(self, mmap_mode: str) -> None:
        indices: dict[str, np.ndarray] = {}
        for name in self.obj_names:
            path = self._index_path(name)
            try:
                array = np.load(path, mmap_mode=mmap_mode, allow_pickle=False)
            except Exception as exc:
                raise ValueError(f"Cannot open KNN cache array {path}: {exc}") from exc
            expected = (
                self.sample_count,
                self.vertex_counts[name],
                self.effective_k[name],
            )
            if array.shape != expected or array.dtype != self.index_dtype:
                raise ValueError(
                    f"KNN cache array mismatch for {name}: expected "
                    f"{expected}/{self.index_dtype}, got {array.shape}/{array.dtype}"
                )
            indices[name] = array
        try:
            completed = np.load(
                self.completed_path, mmap_mode=mmap_mode, allow_pickle=False
            )
            checksums = np.load(
                self.checksums_path, mmap_mode=mmap_mode, allow_pickle=False
            )
        except Exception as exc:
            raise ValueError(f"Cannot open KNN cache commit metadata: {exc}") from exc
        if completed.shape != (self.sample_count,) or completed.dtype != np.uint8:
            raise ValueError(f"Invalid KNN completion bitmap: {self.completed_path}")
        if checksums.shape != (self.sample_count, len(self.obj_names)) or checksums.dtype != np.uint32:
            raise ValueError(f"Invalid KNN checksum array: {self.checksums_path}")
        self._indices = indices
        self._completed = completed
        self._checksums = checksums

    def _validate_file_sizes(self, manifest: Mapping[str, Any]) -> None:
        sizes = manifest.get("file_sizes")
        if not isinstance(sizes, dict):
            raise ValueError(f"Completed KNN cache has no file-size manifest: {self.path}")
        paths = [
            *(self._index_path(name) for name in self.obj_names),
            self.completed_path,
            self.checksums_path,
        ]
        for path in paths:
            expected = sizes.get(path.name)
            try:
                actual = path.stat().st_size
            except FileNotFoundError as exc:
                raise ValueError(f"KNN cache is missing {path}") from exc
            if expected != actual:
                raise ValueError(
                    f"KNN cache file size mismatch for {path.name}: "
                    f"expected {expected}, got {actual}"
                )

    def open(self) -> str:
        self.close()
        if self.final_manifest_path.exists():
            manifest = self._read_json(self.final_manifest_path)
            self._validate_manifest(manifest, "complete")
            self._validate_file_sizes(manifest)
            state = "complete"
        else:
            raise FileNotFoundError(f"KNN aggregate cache is not available: {self.path}")
        self._open_arrays("r")
        if not np.all(self._completed == 1):
            raise ValueError(f"Completed KNN cache has unset rows: {self.path}")
        self._opened_state = state
        return state

    def close(self) -> None:
        self._indices = None
        self._completed = None
        self._checksums = None
        self._opened_state = None

    @property
    def is_complete(self) -> bool:
        if not self.final_manifest_path.exists():
            return False
        try:
            self.open()
        except (OSError, ValueError):
            self.close()
            return False
        self.close()
        return True

    def load(self, sample_index: int, *, verify_checksum: bool = True) -> list[np.ndarray]:
        if not 0 <= sample_index < self.sample_count:
            raise IndexError(sample_index)
        if self._indices is None:
            self.open()
        assert self._indices is not None
        assert self._completed is not None
        assert self._checksums is not None
        if int(self._completed[sample_index]) != 1:
            raise KeyError(f"KNN cache row {sample_index} is not committed")
        rows = []
        for object_index, name in enumerate(self.obj_names):
            row = np.asarray(self._indices[name][sample_index])
            if verify_checksum and _crc32(row) != int(
                self._checksums[sample_index, object_index]
            ):
                raise ValueError(
                    f"KNN cache checksum mismatch at sample {sample_index}, object {name}"
                )
            rows.append(row)
        return rows
