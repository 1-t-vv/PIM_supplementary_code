import os
import json
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.amp import autocast

from model.transformer import Predictor, verify_flash_attention_support
from utils.mesh_dataset_cached import MeshDatasetCached


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

SCENE_CONFIGS = {
    "flat": os.path.join("model", "multi_sphere_flat", "model_config_fine.json"),
    "bumpy": os.path.join("model", "multi_sphere_bumpy", "model_config_fine.json"),
    "jenga": os.path.join("model", "jenga_tower", "model_config_fine.json"),
    "deformable": os.path.join("model", "deformable_box", "model_config_fine.json"),
}


def resolve_config_path(scene: str, config_path: str | None = None) -> str:
    if config_path:
        return config_path
    scene_key = str(scene).strip().lower()
    if scene_key not in SCENE_CONFIGS:
        raise ValueError(f"Unknown scene '{scene}'. Choose one of: {', '.join(sorted(SCENE_CONFIGS))}")
    return SCENE_CONFIGS[scene_key]


def load_model_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_state_dict_flex(ckpt: dict) -> dict:
    for key in ["model", "state_dict", "ema", "module", "net"]:
        if key in ckpt and isinstance(ckpt[key], dict):
            sd = ckpt[key]
            break
    else:
        if isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            sd = ckpt
        else:
            raise RuntimeError("Unexpected checkpoint format; no state_dict-like key found.")

    clean = {}
    for k, v in sd.items():
        if k.startswith("module."):
            k = k[len("module."):]
        clean[k] = v
    return clean


def edge_error_batch(
    Xref_list: list[torch.Tensor],
    Xhat_list: list[torch.Tensor],
    idx_local_list: list[torch.Tensor | None],
    chunk: int = 2048,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Per-sample edge error against GT:
      mean | ||Xhat_p - Xhat_q||_2 - ||Xref_p - Xref_q||_2 |
    over all objects, vertices, and cached local KNN edges.
    """
    if len(Xref_list) != len(Xhat_list) or len(Xref_list) != len(idx_local_list):
        raise ValueError(
            f"List length mismatch: len(Xref)={len(Xref_list)}, "
            f"len(Xhat)={len(Xhat_list)}, len(idx_local)={len(idx_local_list)}"
        )

    err_sum = None
    denom = 0

    for Xref, Xhat, idx_local in zip(Xref_list, Xhat_list, idx_local_list):
        if idx_local is None:
            raise RuntimeError("[Eval] idx_local is None; cannot compute edge error.")
        if idx_local.dtype != torch.long:
            idx_local = idx_local.long()
        if idx_local.ndim == 2:
            idx_local = idx_local.unsqueeze(0).expand(Xref.size(0), -1, -1)

        B, V, _ = Xref.shape
        Xrf = Xref.float()
        Xhf = Xhat.float()
        bidx = torch.arange(B, device=Xref.device)[:, None, None]

        obj_sum = Xref.new_zeros(B, dtype=torch.float32)
        obj_denom = 0
        for s in range(0, V, chunk):
            e = min(V, s + chunk)
            idx = idx_local[:, s:e, :]

            ref_p = Xrf[:, s:e, :].unsqueeze(2)
            hat_p = Xhf[:, s:e, :].unsqueeze(2)
            ref_q = Xrf[bidx, idx]
            hat_q = Xhf[bidx, idx]

            d_ref = (ref_p - ref_q).square().sum(dim=-1).add(eps).sqrt()
            d_hat = (hat_p - hat_q).square().sum(dim=-1).add(eps).sqrt()
            term = (d_hat - d_ref).abs()

            obj_sum = obj_sum + term.reshape(B, -1).sum(dim=1)
            obj_denom += term.numel() // B

        err_sum = obj_sum if err_sum is None else err_sum + obj_sum
        denom += obj_denom

    if err_sum is None:
        return Xhat_list[0].new_zeros(Xhat_list[0].size(0))
    return err_sum / max(denom, 1)


def evaluate(
    ckpt_path: str | None = None,
    test_data_path: str | None = None,
    batch_size: int = 128,
    num_workers: int = 8,
    results_dir: str | None = None,
    center_weight: float = 0.0,
    scene: str = "flat",
    model_config_path: str | None = None,
):
    model_config_path = resolve_config_path(scene, model_config_path)
    file_cfg = load_model_config(model_config_path)
    scene_name = str(file_cfg.get("scene_name", scene)).strip().lower()

    if ckpt_path is None:
        ckpt_path = os.path.join(
            file_cfg.get("save_dir", f"out/multi_sphere_{scene_name}/checkpoint/fine"),
            "best_validate_error_model.pth",
        )
    if test_data_path is None:
        test_data_path = file_cfg.get(
            "test_npz",
            f"data/multi_sphere_{scene_name}/preprocessed/fine/test.npz",
        )
    if results_dir is None:
        results_dir = file_cfg.get("results_dir", os.path.join("out", f"multi_sphere_{scene_name}", "results"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Device: {device}")
    if device.type == "cuda":
        print(f"[Eval] GPU: {torch.cuda.get_device_name(0)}")
        print("[Eval] Attention backend: FlashAttention (forced; no SDPA fallback)")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        print("[Eval] Attention backend: SDPA math (CPU compatibility mode)")

    if not os.path.exists(ckpt_path):
        legacy_ckpt_paths = []
        if ckpt_path.endswith("best_validate_error_model.pth"):
            legacy_ckpt_paths.extend([
                ckpt_path.replace("best_validate_error_model.pth", "best_validate_loss_model.pth"),
                ckpt_path.replace("best_validate_error_model.pth", "best_eval_loss_model.pth"),
            ])
        elif ckpt_path.endswith("best_validate_loss_model.pth"):
            legacy_ckpt_paths.append(ckpt_path.replace("best_validate_loss_model.pth", "best_eval_loss_model.pth"))
        for legacy_ckpt_path in legacy_ckpt_paths:
            if os.path.exists(legacy_ckpt_path):
                print(f"[Eval] Checkpoint not found; using legacy checkpoint: {legacy_ckpt_path}")
                ckpt_path = legacy_ckpt_path
                break

    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = _load_state_dict_flex(ckpt)

    cfg = ckpt.get("config", {})
    if not isinstance(cfg, dict):
        cfg = {}
    cfg = {**file_cfg, **cfg}

    d_model = int(cfg.get("d_model", 512))
    nhead = int(cfg.get("nhead", 8))
    pre_layers = int(cfg.get("pre_layers", 4))
    post_layers = int(cfg.get("post_layers", 4))
    ff_mult = int(cfg.get("ff_mult", 4))
    dropout = float(cfg.get("dropout", 0.1))
    knn_k = int(cfg.get("knn_k", 32))
    num_anchors = int(cfg.get("num_anchors", knn_k))
    edge_knn_k = int(cfg.get("edge_knn_k", knn_k))
    if edge_knn_k <= 0:
        raise ValueError(f"edge_knn_k must be positive, got {edge_knn_k}")
    cache_knn_k = max(knn_k, edge_knn_k)
    knn_chunk = int(cfg.get("knn_chunk", 1024))
    local_query_chunk = int(cfg.get("local_query_chunk", 256))
    cross_pair_chunk = int(cfg.get("cross_pair_chunk", 8))
    droppath_rate = float(cfg.get("droppath_rate", 0.10))
    gamma_init = float(cfg.get("gamma_init", 1e-3))
    zero_init_residual = bool(cfg.get("zero_init_residual", False))
    zero_init_decoder = bool(cfg.get("zero_init_decoder", True))

    if device.type == "cuda":
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)
        verify_flash_attention_support(device, head_dim=d_model // nhead, key_length=knn_k)
        verify_flash_attention_support(device, head_dim=d_model // nhead, key_length=num_anchors)

    print("[Eval] Model hyper-parameters:")
    print(json.dumps({
        "d_model": d_model,
        "nhead": nhead,
        "pre_layers": pre_layers,
        "post_layers": post_layers,
        "ff_mult": ff_mult,
        "dropout": dropout,
        "knn_k": knn_k,
        "num_anchors": num_anchors,
        "edge_knn_k": edge_knn_k,
        "knn_chunk": knn_chunk,
        "local_query_chunk": local_query_chunk,
        "cross_pair_chunk": cross_pair_chunk,
        "droppath_rate": droppath_rate,
        "gamma_init": gamma_init,
        "zero_init_residual": zero_init_residual,
        "zero_init_decoder": zero_init_decoder,
    }, indent=2, ensure_ascii=False))

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

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    epoch = ckpt.get("epoch", None)
    best_error = ckpt.get(
        "best_validate_l1",
        ckpt.get("best_eval_l1", ckpt.get("best_eval_loss", ckpt.get("best_train_loss", None))),
    )
    eval_split = ckpt.get("eval_split", "validate" if "best_validate_l1" in ckpt else "eval")
    if epoch is not None or best_error is not None:
        print(f"[Eval] Loaded ckpt (epoch={None if epoch is None else epoch+1}, best_{eval_split}={best_error})")
    else:
        print("[Eval] Loaded ckpt.")

    idx_cache_dir_test = file_cfg.get(
        "idx_cache_dir_test",
        cfg.get("idx_cache_dir_test", f"data/multi_sphere_{scene_name}/idx_cache/fine_test"),
    )

    test_dataset = MeshDatasetCached(
        npz_path=test_data_path,
        cache_dir=idx_cache_dir_test,
        k=cache_knn_k,
        chunk=knn_chunk,
        knn_cache_format=str(cfg.get("knn_cache_format", "aggregate")),
    )

    test_dataset.require_complete_cache()

    obj_names = list(test_dataset.obj_names)
    M = len(obj_names)
    print(f"[Eval] Objects: {obj_names}")

    if getattr(test_dataset, "scene_ids_np", None) is not None:
        scene_ids = [str(s) for s in np.asarray(test_dataset.scene_ids_np).tolist()]
    else:
        scene_ids = [f"{i:07d}" for i in range(len(test_dataset))]

    print(f"[Eval] Test samples: {len(test_dataset)}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    sum_vertex = 0.0
    sum_center = 0.0
    sum_edge = 0.0
    num_samples = 0

    per_scene_vertex = []
    per_scene_center = []
    per_scene_edge = []

    preds_per_obj = {name: [] for name in obj_names}

    os.makedirs(results_dir, exist_ok=True)

    use_amp = (device.type == "cuda")
    with torch.no_grad(), autocast(device_type="cuda", enabled=use_amp):
        for batch in tqdm(test_loader, desc="Evaluating"):
            (X0_list,
             v0_list,
             f0_list,
             rho0_list,
             fric_list,
             ft, fspin, froll, gz,
             Y_list,
             idx_local_list) = batch

            X0_list   = [x.to(device, non_blocking=True) for x in X0_list]
            v0_list   = [v.to(device, non_blocking=True) for v in v0_list]
            f0_list   = [f.to(device, non_blocking=True) for f in f0_list]
            rho0_list = [rho.to(device, non_blocking=True) for rho in rho0_list]
            fric_list = [fr.to(device, non_blocking=True) for fr in fric_list]
            Y_list    = [y.to(device, non_blocking=True) for y in Y_list]

            idx_local_list = [
                (idx.to(device, non_blocking=True) if isinstance(idx, torch.Tensor) else None)
                for idx in idx_local_list
            ]
            idx_local_for_edge = [
                (idx[..., :edge_knn_k] if isinstance(idx, torch.Tensor) else None)
                for idx in idx_local_list
            ]

            ft    = ft.to(device, non_blocking=True)
            fspin = fspin.to(device, non_blocking=True)
            froll = froll.to(device, non_blocking=True)
            gz    = gz.to(device, non_blocking=True)

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

            diff_list = []
            for X_hat, Y in zip(X_hat_list, Y_list):
                diff_list.append(X_hat - Y)  # (B, V_i, 3)

            diff_all = torch.cat(diff_list, dim=1)  # (B, sum_i V_i, 3)
            vertex_batch = diff_all.float().norm(dim=-1).mean(dim=1)  # (B,)

            center_errs = []
            for X_hat, Y in zip(X_hat_list, Y_list):
                center_pred = X_hat.mean(dim=1)  # (B,3)
                center_gt   = Y.mean(dim=1)      # (B,3)
                center_errs.append((center_pred - center_gt).float().norm(dim=-1))  # (B,)
            center_batch = torch.stack(center_errs, dim=0).mean(dim=0)  # (B,)

            edge_batch = edge_error_batch(
                Xref_list=Y_list,
                Xhat_list=X_hat_list,
                idx_local_list=idx_local_for_edge,
                chunk=knn_chunk,
            )

            sum_vertex += vertex_batch.sum().item()
            sum_center += center_batch.sum().item()
            sum_edge += edge_batch.sum().item()
            B = X0_list[0].size(0)
            num_samples += B

            per_scene_vertex.extend(vertex_batch.detach().cpu().tolist())
            per_scene_center.extend(center_batch.detach().cpu().tolist())
            per_scene_edge.extend(edge_batch.detach().cpu().tolist())

            for name, X_hat in zip(obj_names, X_hat_list):
                preds_per_obj[name].append(X_hat.detach().cpu().numpy())  # (B, V_i, 3)

    mean_vertex = sum_vertex / max(num_samples, 1)
    mean_center = sum_center / max(num_samples, 1)
    mean_edge = sum_edge / max(num_samples, 1)

    print(
        f"[Eval] Vertex error (L2): {mean_vertex:.6f} | "
        f"Center error (L2): {mean_center:.6f} | "
        f"Edge error (vs GT): {mean_edge:.6f}"
    )

    for name in obj_names:
        preds_per_obj[name] = np.concatenate(preds_per_obj[name], axis=0).astype(np.float32)

    pred_npz_path = os.path.join(results_dir, "predictions_multi.npz")
    npz_kwargs = {
        "obj_names": np.asarray(obj_names),
        "scene_ids": np.asarray(scene_ids),
    }
    for name in obj_names:
        npz_kwargs[f"{name}_pred"] = preds_per_obj[name]
    np.savez(pred_npz_path, **npz_kwargs)
    print(f"[Eval] Predictions saved to: {pred_npz_path}")

    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "vertex_error_l2": float(mean_vertex),
            "center_error_l2": float(mean_center),
            "edge_error": float(mean_edge),
            "edge_error_reference": "GT final mesh",
            "edge_knn_k": edge_knn_k,
            "num_samples": int(num_samples),
            "obj_names": obj_names,
        }, f, indent=2)
    print(f"[Eval] Metrics saved to: {metrics_path}")

    error_txt_path = os.path.join(results_dir, "errors.txt")
    with open(error_txt_path, "w", encoding="utf-8") as f:
        f.write("# Per-test-sample evaluation metrics\n")
        f.write("# VertexErrorL2 = mean per-vertex L2 distance to GT final vertices\n")
        f.write("# CenterErrorL2 = mean per-object center L2 distance to GT final centers\n")
        f.write("# EdgeError = mean local KNN edge-length error against GT final mesh\n")
        f.write("index\tscene_id\tVertexErrorL2\tCenterErrorL2\tEdgeError\n")
        for i in range(num_samples):
            sid = scene_ids[i] if i < len(scene_ids) else f"{i:07d}"
            f.write(
                f"{i}\t{sid}\t"
                f"{per_scene_vertex[i]:.10f}\t"
                f"{per_scene_center[i]:.10f}\t"
                f"{per_scene_edge[i]:.10f}\n"
            )
    print(f"[Eval] Per-test errors saved to: {error_txt_path}")

    print("[Eval] Writing per-sample .txt files ...")
    N = num_samples
    for i in tqdm(range(N), desc="Saving .txt"):
        sid = scene_ids[i] if i < len(scene_ids) else f"{i:07d}"
        out_txt = os.path.join(results_dir, f"{sid}.txt")
        with open(out_txt, "w") as f:
            for name in obj_names:
                verts = preds_per_obj[name][i]  # (V_i, 3)
                f.write(f"# {name}_Final\n")
                for v in verts:
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

    print(f"[Eval] Done. Results in: {results_dir}")
    return {
        "vertex_error_l2": float(mean_vertex),
        "center_error_l2": float(mean_center),
        "edge_error": float(mean_edge),
    }


if __name__ == "__main__":
    os.chdir(PROJECT_DIR)
    parser = argparse.ArgumentParser(description="Evaluate a selected scene on its test split.")
    parser.add_argument("--scene", choices=sorted(SCENE_CONFIGS), default="flat")
    parser.add_argument("--config", type=str, default=None, help="Optional explicit fine model config path.")
    parser.add_argument("--ckpt", type=str, default=None, help="Optional checkpoint path.")
    parser.add_argument("--test-data", type=str, default=None, help="Optional test .store or legacy .npz path.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--center-weight", type=float, default=0.0, help="Deprecated; kept for CLI compatibility.")
    args = parser.parse_args()

    evaluate(
        ckpt_path=args.ckpt,
        test_data_path=args.test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        results_dir=args.results_dir,
        center_weight=args.center_weight,
        scene=args.scene,
        model_config_path=args.config,
    )
