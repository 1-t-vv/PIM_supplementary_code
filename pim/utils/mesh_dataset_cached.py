import os
import json
from typing import List, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.array_store import load_dataset_arrays
from utils.knn_cache_store import KnnCacheStore
from utils.knn_cache_store import KNN_ALGORITHM

class MeshDatasetCached(Dataset):

    def __init__(
        self,
        npz_path: str,
        cache_dir: str,
        k: int = 32,
        chunk: int = 1024,
        use_knn_cache: bool = True,
        knn_cache_format: str = "aggregate",
        verify_data_checksums: bool = False,
        verify_knn_checksums: bool = True,
    ):
        super().__init__()
        self._reopen_args = {
            "npz_path": npz_path,
            "cache_dir": cache_dir,
            "k": k,
            "chunk": chunk,
            "use_knn_cache": use_knn_cache,
            "knn_cache_format": knn_cache_format,
            "verify_data_checksums": verify_data_checksums,
            "verify_knn_checksums": verify_knn_checksums,
        }
        arrays, data_manifest, resolved_path = load_dataset_arrays(
            npz_path, verify_checksums=verify_data_checksums
        )
        self.data_path = str(resolved_path)
        self._reopen_args["npz_path"] = str(resolved_path.resolve())
        self._reopen_args["cache_dir"] = str(os.path.abspath(cache_dir))
        self.dataset_fingerprint = str(
            data_manifest.get(
                "cache_fingerprint", data_manifest["dataset_fingerprint"]
            )
        )
        self.data_content_fingerprint = str(data_manifest["dataset_fingerprint"])
        self.data_storage_format = str(data_manifest["format"])

        if "obj_names" not in arrays:
            raise ValueError(f"{resolved_path} missing 'obj_names'")

        self.obj_names: List[str] = [str(x) for x in np.asarray(arrays["obj_names"]).tolist()]
        self.M = len(self.obj_names)
        if self.M == 0 or len(set(self.obj_names)) != self.M:
            raise ValueError(f"{resolved_path} has empty or duplicate obj_names")

        global_keys = ["gravity_z", "friction_t", "friction_spin", "friction_roll"]
        if not all(key in arrays for key in global_keys):
            raise ValueError(f"{resolved_path} missing some of global keys {global_keys}")

        self.gravity_z_np = np.asarray(arrays["gravity_z"], dtype=np.float32).reshape(-1)
        self.friction_t_np = np.asarray(arrays["friction_t"], dtype=np.float32).reshape(-1)
        self.friction_spin_np = np.asarray(arrays["friction_spin"], dtype=np.float32).reshape(-1)
        self.friction_roll_np = np.asarray(arrays["friction_roll"], dtype=np.float32).reshape(-1)
        self.scene_ids_np = np.asarray(arrays["scene_ids"]) if "scene_ids" in arrays else None

        self.N = self.gravity_z_np.shape[0]
        if self.N <= 0:
            raise ValueError(f"{resolved_path} contains no samples")
        if self.scene_ids_np is not None and self.scene_ids_np.reshape(-1).shape != (self.N,):
            raise ValueError(
                f"{resolved_path} scene_ids has inconsistent sample count: "
                f"{self.scene_ids_np.shape} vs N={self.N}"
            )
        for key, value in (
            ("friction_t", self.friction_t_np),
            ("friction_spin", self.friction_spin_np),
            ("friction_roll", self.friction_roll_np),
        ):
            if value.shape != (self.N,):
                raise ValueError(f"{resolved_path} field {key} has inconsistent sample count")

        self.X_init_np: Dict[str, np.ndarray] = {}
        self.X_final_np: Dict[str, np.ndarray] = {}
        self.vel0_np: Dict[str, np.ndarray] = {}
        self.force0_np: Dict[str, np.ndarray] = {}
        self.density_np: Dict[str, np.ndarray] = {}
        self.friction_np: Dict[str, np.ndarray] = {}

        self.pos0_np: Dict[str, np.ndarray] = {}
        self.material_np: Dict[str, np.ndarray] = {}
        self.faces_np: Dict[str, np.ndarray] = {}

        for name in self.obj_names:
            key_init = f"{name}_init"
            key_final = f"{name}_final"
            key_vel0 = f"{name}_vel0"
            key_force0 = f"{name}_force0"
            key_dens = f"{name}_density"
            key_fric = f"{name}_friction"

            for k_req in [key_init, key_final, key_vel0, key_force0, key_dens, key_fric]:
                if k_req not in arrays:
                    raise ValueError(f"{resolved_path} missing field '{k_req}'")

            X_init = np.asarray(arrays[key_init], dtype=np.float32)
            X_final = np.asarray(arrays[key_final], dtype=np.float32)
            if X_init.shape != X_final.shape:
                raise ValueError(f"{key_init} / {key_final} shape mismatch: {X_init.shape} vs {X_final.shape}")
            if X_init.ndim != 3 or X_init.shape[-1] != 3:
                raise ValueError(f"{key_init} must be (N, V, 3), got {X_init.shape}")
            if X_init.shape[0] != self.N or X_init.shape[1] <= 0:
                raise ValueError(
                    f"{key_init} must contain N={self.N} non-empty meshes, got {X_init.shape}"
                )

            vel0 = np.asarray(arrays[key_vel0], dtype=np.float32)
            force0 = np.asarray(arrays[key_force0], dtype=np.float32)
            if vel0.shape != (self.N, 3) or force0.shape != (self.N, 3):
                raise ValueError(f"{key_vel0}/{key_force0} must be (N,3), got {vel0.shape}/{force0.shape}")

            dens = np.asarray(arrays[key_dens], dtype=np.float32)
            if dens.ndim == 1:
                dens = dens[:, None]
            elif dens.ndim == 2 and dens.shape[1] == 1:
                pass
            else:
                raise ValueError(f"{key_dens} must be (N,) or (N,1), got {dens.shape}")
            if dens.shape[0] != self.N:
                raise ValueError(f"{key_dens} N mismatch: {dens.shape[0]} vs {self.N}")

            fric = np.asarray(arrays[key_fric], dtype=np.float32)
            if fric.shape != (self.N, 3):
                raise ValueError(f"{key_fric} must be (N,3), got {fric.shape}")

            self.X_init_np[name] = X_init
            self.X_final_np[name] = X_final
            self.vel0_np[name] = vel0
            self.force0_np[name] = force0
            self.density_np[name] = dens
            self.friction_np[name] = fric

            key_pos0 = f"{name}_pos0"
            if key_pos0 in arrays:
                self.pos0_np[name] = np.asarray(arrays[key_pos0], dtype=np.float32)

            key_material = f"{name}_material"
            if key_material in arrays:
                self.material_np[name] = np.asarray(arrays[key_material])

            key_faces = f"{name}_faces"
            if key_faces in arrays:
                self.faces_np[name] = np.asarray(arrays[key_faces], dtype=np.int32)

        self.k = int(k)
        self.chunk = int(chunk)

        self.use_knn_cache = bool(use_knn_cache)
        self.knn_cache_format = str(knn_cache_format).strip().lower()
        if self.knn_cache_format not in {"aggregate", "legacy"}:
            raise ValueError(
                f"knn_cache_format must be 'aggregate' or 'legacy', got {knn_cache_format!r}"
            )
        self.verify_knn_checksums = bool(verify_knn_checksums)

        self.cache_dir = cache_dir

        max_vert = 0
        for name in self.obj_names:
            V = self.X_init_np[name].shape[1]
            if V > max_vert:
                max_vert = V
        self._np_index_dtype = np.dtype(
            np.uint16 if max_vert <= np.iinfo(np.uint16).max + 1 else np.uint32
        )
        if max_vert > np.iinfo(np.uint32).max + 1:
            raise ValueError(f"Vertex count {max_vert} exceeds uint32 KNN index capacity")
        self._knn_store: Optional[KnnCacheStore] = None
        self._verified_knn_rows: set[int] = set()
        self._trusted_legacy_cache = self._legacy_cache_has_matching_provenance()
        if self.use_knn_cache and self.knn_cache_format == "aggregate":
            self._knn_store = KnnCacheStore(
                cache_dir,
                dataset_fingerprint=self.dataset_fingerprint,
                obj_names=self.obj_names,
                sample_count=self.N,
                vertex_counts={
                    name: int(self.X_init_np[name].shape[1])
                    for name in self.obj_names
                },
                k=self.k,
                index_dtype=self._np_index_dtype,
            )

    def __len__(self) -> int:
        return self.N

    def __getstate__(self):
        if self.data_storage_format != "mesh-array-store":
            return self.__dict__
        # Reopen mmap files from their manifest in spawn-based DataLoader
        # workers. This avoids serializing multi-gigabyte ndarray views.
        return {"_array_store_reopen_args": self._reopen_args}

    def __setstate__(self, state):
        reopen_args = state.get("_array_store_reopen_args")
        if reopen_args is None:
            self.__dict__.update(state)
            return
        self.__init__(**reopen_args)

    def _cache_path(self, i: int) -> str:
        return os.path.join(self.cache_dir, f"{i:07d}.npz")

    def _legacy_cache_has_matching_provenance(self) -> bool:
        marker = os.path.join(self.cache_dir, ".cache_ready")
        try:
            with open(marker, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except (OSError, json.JSONDecodeError):
            return False
        return (
            metadata.get("ready") is True
            and metadata.get("format") == "legacy"
            and metadata.get("dataset_fingerprint") == self.dataset_fingerprint
            and metadata.get("requested_k") == self.k
            and metadata.get("algorithm") == KNN_ALGORITHM
            and metadata.get("sample_count") == self.N
            and metadata.get("obj_names") == self.obj_names
        )

    def require_complete_cache(self) -> None:
        """Validate that all packaged KNN rows are present without creating data."""
        if not self.use_knn_cache:
            return
        if self.knn_cache_format == "aggregate":
            assert self._knn_store is not None
            self._knn_store.open()
            self._knn_store.close()
            return
        if not self._trusted_legacy_cache:
            raise ValueError(f"Legacy KNN cache provenance is invalid: {self.cache_dir}")
        missing = [i for i in range(self.N) if not os.path.isfile(self._cache_path(i))]
        if missing:
            raise FileNotFoundError(
                f"Legacy KNN cache is incomplete in {self.cache_dir}; missing row {missing[0]}"
            )

    def _load_cached_indices(
        self,
        i: int
    ) -> List[torch.Tensor]:
        if self.knn_cache_format == "aggregate":
            assert self._knn_store is not None
            verify_row = self.verify_knn_checksums and i not in self._verified_knn_rows
            rows = self._knn_store.load(i, verify_checksum=verify_row)
            if verify_row:
                self._verified_knn_rows.add(i)
            return [
                torch.from_numpy(np.asarray(row).astype(np.int64, copy=True))
                for row in rows
            ]

        p = self._cache_path(i)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Packaged KNN cache row is missing: {p}")

        # Read only local arrays. np.load keeps compressed members lazy, so
        # legacy idx_*_to_* members are neither decompressed nor materialized.
        idx_local_list: List[torch.Tensor] = []
        with np.load(p) as dat:
            for name in self.obj_names:
                key = f"idx_{name}"
                if key not in dat.files:
                    raise ValueError(f"cache file {p} missing '{key}'")
                idx_np = dat[key].astype(np.int64)  # (V,k)
                idx_local_list.append(torch.from_numpy(idx_np))

        # Old cache files may contain idx_*_to_* arrays. They are intentionally
        # ignored: cross-attention anchors are now generated inside the model.
        return idx_local_list

    @property
    def knn_cache_path(self) -> Optional[str]:
        if self._knn_store is not None:
            return str(self._knn_store.path)
        return self.cache_dir if self.use_knn_cache else None

    def __getitem__(self, i: int):
        X0_list: List[torch.Tensor] = []
        Y_list: List[torch.Tensor] = []
        v0_list: List[torch.Tensor] = []
        f0_list: List[torch.Tensor] = []
        rho0_list: List[torch.Tensor] = []
        fric_list: List[torch.Tensor] = []

        for name in self.obj_names:
            # mmap-backed arrays are read-only. Copy only the requested sample
            # into writable CPU memory before exposing it to PyTorch.
            X0 = torch.from_numpy(np.array(self.X_init_np[name][i], dtype=np.float32, copy=True))
            Y = torch.from_numpy(np.array(self.X_final_np[name][i], dtype=np.float32, copy=True))
            v0 = torch.from_numpy(np.array(self.vel0_np[name][i], dtype=np.float32, copy=True))
            f0 = torch.from_numpy(np.array(self.force0_np[name][i], dtype=np.float32, copy=True))
            rho = torch.from_numpy(np.array(self.density_np[name][i], dtype=np.float32, copy=True))
            fr = torch.from_numpy(np.array(self.friction_np[name][i], dtype=np.float32, copy=True))

            X0_list.append(X0)
            Y_list.append(Y)
            v0_list.append(v0)
            f0_list.append(f0)
            rho0_list.append(rho)
            fric_list.append(fr)

        if self.use_knn_cache:
            idx_local_list = self._load_cached_indices(i)
        else:
            idx_local_list = []
            for name in self.obj_names:
                V = self.X_init_np[name].shape[1]     # (N,V,3) -> V
                idx_local_list.append(torch.empty((V, 0), dtype=torch.long))

        ft    = torch.tensor([self.friction_t_np[i]],    dtype=torch.float32)  # (1,)
        fspin = torch.tensor([self.friction_spin_np[i]], dtype=torch.float32)  # (1,)
        froll = torch.tensor([self.friction_roll_np[i]], dtype=torch.float32)  # (1,)
        gz    = torch.tensor([self.gravity_z_np[i]],     dtype=torch.float32)  # (1,)

        return (
            X0_list,
            v0_list,
            f0_list,
            rho0_list,
            fric_list,
            ft, fspin, froll, gz,
            Y_list,
            idx_local_list,
        )
