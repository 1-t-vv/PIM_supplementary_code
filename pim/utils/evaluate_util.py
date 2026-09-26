# evaluate.py
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader

from utils.mesh_dataset_cached import MeshDatasetCached
from utils.array_store import resolve_dataset_path


def center_loss_multi(
    Xhat_list: List[torch.Tensor],
    Y_list: List[torch.Tensor],
    mode: str = "euclidean_sq",
) -> torch.Tensor:
    cen_sum = None
    denom = 0

    for Xhat, Y in zip(Xhat_list, Y_list):
        chat = Xhat.mean(dim=1)  # (B,3)
        cy   = Y.mean(dim=1)     # (B,3)
        dc   = chat - cy         # (B,3)

        if mode == "euclidean":
            term = dc.norm(dim=-1)                 # (B,)
        elif mode == "euclidean_sq":
            term = dc.square().sum(dim=-1)         # (B,)
        elif mode == "smooth_l1_vec":
            term = F.smooth_l1_loss(chat, cy, reduction="none").sum(dim=-1)  # (B,)
        elif mode == "l1_vec":
            term = dc.abs().sum(dim=-1)            # (B,)
        elif mode == "l2_vec":
            term = dc.square().mean(dim=-1)        # (B,)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        cen_sum = term.sum() if (cen_sum is None) else (cen_sum + term.sum())
        denom += term.numel()

    if cen_sum is None:
        return Xhat_list[0].new_tensor(0.0)
    return cen_sum / max(denom, 1)


def knn_edge_loss_multi(
    Xref_list: List[torch.Tensor],
    Xhat_list: List[torch.Tensor],
    idx_local_list: List[Optional[torch.Tensor]],
    mode: str = "l1",
    chunk: int = 2048,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Edge loss based on cached kNN (idx_local):
      match per-edge (i -> knn(i)) distances between Xref and Xhat.

    Xref: (B,V,3), usually GT final vertices
    Xhat: (B,V,3), predicted vertices
    idx:  (B,V,k) long
    """
    if len(Xref_list) != len(Xhat_list) or len(Xref_list) != len(idx_local_list):
        raise ValueError(
            f"List length mismatch: len(Xref)={len(Xref_list)}, len(Xhat)={len(Xhat_list)}, len(idx_local)={len(idx_local_list)}"
        )

    loss_sum = None
    denom = 0

    for Xref, Xhat, idx_local in zip(Xref_list, Xhat_list, idx_local_list):
        if idx_local is None:
            raise RuntimeError("[EdgeLoss] idx_local is None. Expected cached per-object idx_local.")

        if idx_local.dtype != torch.long:
            idx_local = idx_local.long()

        B, V, _ = Xref.shape
        # float32 for stability under AMP
        Xrf = Xref.float()
        Xhf = Xhat.float()

        # batch index helper for advanced indexing
        bidx = torch.arange(B, device=Xref.device)[:, None, None]

        # process points in chunks along V to limit peak memory
        for s in range(0, V, chunk):
            e = min(V, s + chunk)

            idx = idx_local[:, s:e, :]                    # (B,vc,k)
            p0  = Xrf[:, s:e, :].unsqueeze(2)             # (B,vc,1,3)
            p1  = Xhf[:, s:e, :].unsqueeze(2)             # (B,vc,1,3)

            n0 = Xrf[bidx, idx]                           # (B,vc,k,3)
            n1 = Xhf[bidx, idx]                           # (B,vc,k,3)

            # Euclidean distances
            d0 = (p0 - n0).square().sum(dim=-1).add(eps).sqrt()  # (B,vc,k)
            d1 = (p1 - n1).square().sum(dim=-1).add(eps).sqrt()  # (B,vc,k)

            dd = d1 - d0
            if mode == "l1":
                term = dd.abs()
            elif mode == "l2":
                term = dd.square()
            elif mode == "smooth_l1":
                term = F.smooth_l1_loss(d1, d0, reduction="none")
            else:
                raise ValueError(f"Unknown mode: {mode}")

            loss_sum = term.sum() if (loss_sum is None) else (loss_sum + term.sum())
            denom += term.numel()

    if loss_sum is None:
        return Xhat_list[0].new_tensor(0.0)
    return loss_sum / max(denom, 1)


class ValidationEvaluator:
    """Persistent validation Dataset/DataLoader for in-training evaluation.

    The caller supplies the already-trained model instance. This avoids
    rebuilding a second Predictor, loading a state dict, reopening the dataset,
    and respawning DataLoader workers after every epoch.
    """

    def __init__(
        self,
        *,
        test_data_path: str,
        model_config_path: str,
        cache_dir: Optional[str] = None,
        split_name: str = "validate",
        device: Optional[torch.device] = None,
    ) -> None:
        resolved_test_data_path = resolve_dataset_path(test_data_path)
        with open(model_config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.test_data_path = str(resolved_test_data_path)
        self.split_name = str(split_name or "validate")
        self.edge_mode = str(cfg.get("edge_mode", cfg.get("shape_mode", "l1")))
        self.edge_chunk = int(cfg.get("edge_chunk", cfg.get("shape_chunk", 2048)))

        knn_k = int(cfg.get("knn_k", 32))
        self.edge_knn_k = int(cfg.get("edge_knn_k", knn_k))
        if self.edge_knn_k <= 0:
            raise ValueError(
                f"edge_knn_k must be positive, got {self.edge_knn_k}"
            )
        cache_knn_k = max(knn_k, self.edge_knn_k)
        knn_chunk = int(cfg.get("knn_chunk", 1024))
        use_knn_cache = bool(cfg.get("use_knn_cache", True))
        knn_cache_format = str(cfg.get("knn_cache_format", "aggregate"))
        if cache_dir is None:
            scene_name = str(cfg.get("scene_name", "flat")).strip().lower()
            cache_dir = cfg.get(
                f"idx_cache_dir_{self.split_name}",
                f"data/multi_sphere_{scene_name}/idx_cache/fine_{self.split_name}",
            )

        self.dataset = MeshDatasetCached(
            npz_path=str(resolved_test_data_path),
            cache_dir=cache_dir,
            k=cache_knn_k,
            chunk=knn_chunk,
            use_knn_cache=use_knn_cache,
            knn_cache_format=knn_cache_format,
        )
        if use_knn_cache:
            self.dataset.require_complete_cache()

        num_workers = int(cfg.get("eval_num_workers", 0))
        self.loader = DataLoader(
            self.dataset,
            batch_size=int(cfg.get("eval_batch_size", 16)),
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
            pin_memory=(self.device.type == "cuda"),
            drop_last=False,
        )

    def run(
        self,
        model: torch.nn.Module,
        *,
        ckpt_epoch: Optional[int] = None,
    ) -> Dict[str, Any]:
        was_training = model.training
        model.eval()

        # [abs_sum, abs_count, edge_sum, edge_count,
        #  center_sum, center_count, sample_count]
        totals = torch.zeros(7, device=self.device, dtype=torch.float64)
        try:
            with torch.inference_mode(), autocast(
                device_type="cuda", enabled=(self.device.type == "cuda")
            ):
                for batch in self.loader:
                    (
                        X0_list,
                        v0_list,
                        f0_list,
                        rho0_list,
                        fric_list,
                        ft,
                        fspin,
                        froll,
                        gz,
                        Y_list,
                        idx_local_list,
                    ) = batch

                    X0_list = [x.to(self.device, non_blocking=True) for x in X0_list]
                    v0_list = [v.to(self.device, non_blocking=True) for v in v0_list]
                    f0_list = [f.to(self.device, non_blocking=True) for f in f0_list]
                    rho0_list = [rho.to(self.device, non_blocking=True) for rho in rho0_list]
                    fric_list = [fr.to(self.device, non_blocking=True) for fr in fric_list]
                    Y_list = [y.to(self.device, non_blocking=True) for y in Y_list]
                    idx_local_list = [
                        idx.to(self.device, non_blocking=True) for idx in idx_local_list
                    ]
                    idx_local_for_edge = [
                        idx[..., :self.edge_knn_k] for idx in idx_local_list
                    ]
                    ft = ft.to(self.device, non_blocking=True)
                    fspin = fspin.to(self.device, non_blocking=True)
                    froll = froll.to(self.device, non_blocking=True)
                    gz = gz.to(self.device, non_blocking=True)

                    X_hat_list, aux = model(
                        X0_list=X0_list,
                        v0_list=v0_list,
                        f0_list=f0_list,
                        rho0_list=rho0_list,
                        fric_list=fric_list,
                        ft=ft,
                        fspin=fspin,
                        froll=froll,
                        gz=gz,
                        idx_local_list=idx_local_list,
                    )

                    abs_sum = X_hat_list[0].new_zeros((), dtype=torch.float32)
                    abs_count = 0
                    center_sum = X_hat_list[0].new_zeros((), dtype=torch.float32)
                    center_count = 0
                    for Xhat, Y in zip(X_hat_list, Y_list):
                        abs_sum = abs_sum + (Xhat - Y).float().abs().sum()
                        abs_count += Xhat.numel()
                        dc = (Xhat.mean(dim=1) - Y.mean(dim=1)).float()
                        center_sum = center_sum + dc.square().sum(dim=-1).sum()
                        center_count += dc.size(0)

                    edge_mean = knn_edge_loss_multi(
                        Xref_list=Y_list,
                        Xhat_list=X_hat_list,
                        idx_local_list=idx_local_for_edge,
                        mode=self.edge_mode,
                        chunk=self.edge_chunk,
                    )
                    edge_count = sum(idx.numel() for idx in idx_local_for_edge)
                    batch_size = X0_list[0].size(0)

                    batch_totals = torch.stack(
                        (
                            abs_sum,
                            abs_sum.new_tensor(abs_count),
                            edge_mean * edge_count,
                            abs_sum.new_tensor(edge_count),
                            center_sum,
                            abs_sum.new_tensor(center_count),
                            abs_sum.new_tensor(batch_size),
                        )
                    ).to(torch.float64)
                    totals.add_(batch_totals)

            # One host synchronization/conversion for the entire validation pass.
            (
                sum_abs,
                denom_abs,
                sum_edge,
                denom_edge,
                sum_cen,
                denom_cen,
                num_samples,
            ) = totals.cpu().tolist()
            return {
                "eval_l1": float(sum_abs / max(denom_abs, 1.0)),
                "eval_edge": float(sum_edge / max(denom_edge, 1.0)),
                "eval_cen": float(sum_cen / max(denom_cen, 1.0)),
                "num_samples": int(num_samples),
                "split": self.split_name,
                "data_path": self.test_data_path,
                "epoch": ckpt_epoch,
            }
        finally:
            model.train(was_training)


def evaluate(
    log_file: str = "out/multi_sphere_flat/logs/evaluate_util.log",
    test_data_path: str = "data/multi_sphere_flat/preprocessed/coarse/validate.store",
    ckpt_path: Optional[str] = "out/multi_sphere_flat/checkpoint/fine/best_validate_error_model.pth",
    model_config_path: str = "model/multi_sphere_flat/model_config_fine.json",
    model_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    ckpt_epoch: Optional[int] = None,
    cache_dir: Optional[str] = None,
    split_name: str = "validate",
):
    from model.transformer import Predictor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_state_dict is None and (ckpt_path is None or not os.path.exists(ckpt_path)):
        print(f"[Eval] Checkpoint not found: {ckpt_path}", flush=True)
        return None

    try:
        with open(model_config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:
        print(f"[Eval] Failed to load model config {model_config_path}: {e}", flush=True)
        return None

    knn_k = int(cfg.get("knn_k", 32))
    num_anchors = int(cfg.get("num_anchors", knn_k))
    knn_chunk = int(cfg.get("knn_chunk", 1024))
    local_query_chunk = int(cfg.get("local_query_chunk", 256))
    cross_pair_chunk = int(cfg.get("cross_pair_chunk", 8))

    d_model = int(cfg.get("d_model", 512))
    nhead = int(cfg.get("nhead", 8))
    pre_layers = int(cfg.get("pre_layers", cfg.get("sa1_layers", 4)))
    post_layers = int(cfg.get("post_layers", 4))
    ff_mult = int(cfg.get("ff_mult", 4))
    dropout = float(cfg.get("dropout", 0.1))
    droppath_rate = float(cfg.get("droppath_rate", 0.10))
    gamma_init = float(cfg.get("gamma_init", 1e-3))
    zero_init_residual = bool(cfg.get("zero_init_residual", False))
    zero_init_decoder = bool(cfg.get("zero_init_decoder", True))

    model = Predictor(
        d_model=d_model,
        nhead=nhead,
        pre_layers=pre_layers,
        post_layers=post_layers,
        ff_mult=ff_mult,
        dropout=dropout,
        knn_k=knn_k,
        num_anchors=num_anchors,
        knn_chunk=knn_chunk,
        local_query_chunk=local_query_chunk,
        cross_pair_chunk=cross_pair_chunk,
        droppath_rate=droppath_rate,
        gamma_init=gamma_init,
        zero_init_residual=zero_init_residual,
        zero_init_decoder=zero_init_decoder,
    ).to(device)

    ckpt: Dict[str, Any] = {}
    if model_state_dict is None:
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location=device)

        state = ckpt
        if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        elif isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
    else:
        state = model_state_dict
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}

    try:
        model.load_state_dict(state, strict=True)
    except Exception as e:
        print(f"[Eval] load_state_dict(strict=True) failed: {e}", flush=True)
        return None
    try:
        epoch_for_log = ckpt_epoch
        if epoch_for_log is None and isinstance(ckpt, dict):
            epoch_for_log = ckpt.get("epoch", None)
        evaluator = ValidationEvaluator(
            test_data_path=test_data_path,
            model_config_path=model_config_path,
            cache_dir=cache_dir,
            split_name=split_name,
            device=device,
        )
        return evaluator.run(model, ckpt_epoch=epoch_for_log)

    except Exception as e:
        print(f"[Eval] failed: {e}", flush=True)
        return None
