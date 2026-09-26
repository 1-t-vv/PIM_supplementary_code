import os

os.environ["MKL_THREADING_LAYER"] = "GNU"

import math
import json
import logging
import contextlib
import random
import time
import argparse
from typing import List, Tuple, Dict, Any, Optional, Callable

import numpy as np
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import GradScaler, autocast

from utils.mesh_dataset_cached import MeshDatasetCached
from utils.step_profiler import StepProfiler
from utils.compile_util import compile_training_model
from utils.c4_augmentation import apply_random_c4_rotation
from model.transformer import Predictor, verify_flash_attention_support
from utils.evaluate_util import ValidationEvaluator
from utils.draw_fine import save_plot as save_training_plot


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

SCENE_CONFIGS = {
    "flat": os.path.join("model", "multi_sphere_flat", "model_config_fine.json"),
    "bumpy": os.path.join("model", "multi_sphere_bumpy", "model_config_fine.json"),
    "jenga": os.path.join("model", "jenga_tower", "model_config_fine.json"),
    "deformable": os.path.join("model", "deformable_box", "model_config_fine.json"),
}


def resolve_config_path(scene: str, config_path: Optional[str] = None) -> str:
    if config_path:
        return config_path
    scene_key = str(scene).strip().lower()
    if scene_key not in SCENE_CONFIGS:
        raise ValueError(f"Unknown scene '{scene}'. Choose one of: {', '.join(sorted(SCENE_CONFIGS))}")
    return SCENE_CONFIGS[scene_key]


def setup_logger(rank: int, log_file: str):
    if rank == 0:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

def log_stage(rank: int, message: str):
    text = f"[rank {rank}] {message}"
    print(text, flush=True)

from datetime import timedelta

def setup(rank: int, world_size: int):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "12355")
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=60),   
    )
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def get_cosine_schedule_with_warmup(optimizer,
                                    warmup_steps: int,
                                    total_steps: int,
                                    min_lr: float):
    warmup_steps = int(warmup_steps)
    total_steps = int(total_steps)
    base_lr = optimizer.param_groups[0]["lr"]

    min_ratio = float(min_lr) / float(base_lr)
    min_ratio = max(0.0, min(min_ratio, 1.0))

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


def center_loss_multi(
    Xhat_list,
    Y_list,
    mode: str = "euclidean_sq",  
):
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
            term = dc.square().mean(dim=-1)        # (B,)  (MSE per sample)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        cen_sum = term.sum() if (cen_sum is None) else (cen_sum + term.sum())
        denom += term.numel()

    if cen_sum is None:
        return Xhat_list[0].new_tensor(0.0)
    return cen_sum / max(denom, 1)



import torch
import torch.nn.functional as F
from typing import List, Optional

def knn_edge_loss_multi(
    Xref_list: List[torch.Tensor],
    Xhat_list: List[torch.Tensor],
    idx_local_list: List[Optional[torch.Tensor]],
    mode: str = "l1",   # "l1" | "l2" | "smooth_l1"
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Edge loss via KNN edge-length matching:
      For each object, for each point i and its k cached neighbors j:
        ||Xhat_i - Xhat_j||  should match  ||Xref_i - Xref_j||.

    Args:
      Xref_list: list of (B, V, 3), usually GT final vertices
      Xhat_list: list of (B, V, 3), predicted vertices
      idx_local_list: list of (B, V, k) long tensors
      mode: loss on distance difference

    Returns:
      scalar tensor
    """
    assert len(Xref_list) == len(Xhat_list) == len(idx_local_list)

    loss_sum = None
    denom = 0

    for Xref, Xhat, idx in zip(Xref_list, Xhat_list, idx_local_list):
        if idx is None:
            raise RuntimeError("[EdgeLoss] idx_local is None. You said you want to use cached KNN; please pass it in.")
        if idx.dtype != torch.long:
            idx = idx.long()

        B, V, _ = Xref.shape
        k = idx.size(-1)

        # (B,V,k,3) gather neighbors using advanced indexing
        b = torch.arange(B, device=Xref.device)[:, None, None]  # (B,1,1)
        Xref_nb = Xref[b, idx]   # (B,V,k,3)
        Xhat_nb = Xhat[b, idx]   # (B,V,k,3)

        # distances (use float32 for stability under autocast)
        d0 = (Xref.unsqueeze(2) - Xref_nb).norm(dim=-1).float()  # (B,V,k)
        d1 = (Xhat.unsqueeze(2) - Xhat_nb).norm(dim=-1).float()  # (B,V,k)

        if mode == "l1":
            term = (d1 - d0).abs()
        elif mode == "l2":
            diff = (d1 - d0)
            term = diff * diff
        elif mode == "smooth_l1":
            term = F.smooth_l1_loss(d1, d0, reduction="none")
        else:
            raise ValueError(f"Unknown mode: {mode}")

        loss_sum = term.sum() if (loss_sum is None) else (loss_sum + term.sum())
        denom += term.numel()

    if loss_sum is None:
        return Xref_list[0].new_tensor(0.0)
    return loss_sum / max(denom, 1)



def _coerce_eval_error(x):
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, dict):
            for k in ("eval_l1", "eval_error", "val_error", "error", "eval_loss", "val_loss", "loss", "avg_loss"):
                if k in x:
                    return float(x[k])
        if isinstance(x, (list, tuple)) and len(x) > 0:
            return float(x[0])
    except Exception:
        return None
    return None


def _coerce_metric(x, key: str):
    try:
        if isinstance(x, dict):
            if key in x:
                return float(x[key])
            aliases = {
                "eval_edge": ("eval_rad", "eval_shape"),
                "eval_cen": ("eval_center",),
            }
            for alias in aliases.get(key, ()):
                if alias in x:
                    return float(x[alias])
    except Exception:
        return None
    return None



def load_model_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


def update_training_plot(log_file: str, plot_file: str):
    try:
        for handler in logging.getLogger().handlers:
            handler.flush()
        save_training_plot(log_file, plot_file)
    except Exception as e:
        print(f"[Fine][Plot] failed to update {plot_file}: {e}", flush=True)


def _load_coarse_weights_if_available(
    model: torch.nn.Module,
    device: torch.device,
    cfg: dict,
    rank: int
) -> bool:
    scene_name = str(cfg.get("scene_name", "flat")).strip().lower()
    default_coarse_ckpt = os.path.join(
        "out",
        f"multi_sphere_{scene_name}",
        "checkpoint",
        "coarse",
        "best_validate_error_model.pth",
    )
    coarse_ckpt_path = cfg.get("coarse_ckpt_path", default_coarse_ckpt)
    if (
        "coarse_ckpt_path" not in cfg
        and not os.path.exists(coarse_ckpt_path)
    ):
        for legacy_name in ("best_validate_loss_model.pth", "best_eval_loss_model.pth"):
            legacy_ckpt = os.path.join(
                "out",
                f"multi_sphere_{scene_name}",
                "checkpoint",
                "coarse",
                legacy_name,
            )
            if os.path.exists(legacy_ckpt):
                coarse_ckpt_path = legacy_ckpt
                break

    if not os.path.exists(coarse_ckpt_path):
        if rank == 0:
            print(f"[Fine] Coarse checkpoint not found, training from scratch: {coarse_ckpt_path}", flush=True)
        return False

    if rank == 0:
        print(f"[Fine] Loading coarse checkpoint from: {coarse_ckpt_path}", flush=True)

    try:
        try:
            ckpt = torch.load(coarse_ckpt_path, map_location=device, weights_only=True)
        except TypeError:
            ckpt = torch.load(coarse_ckpt_path, map_location=device)
    except Exception as e:
        if rank == 0:
            print(f"[Fine] Failed to load coarse checkpoint: {e}", flush=True)
        return False

    state = ckpt
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    elif isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}

    try:
        model.load_state_dict(state, strict=True)
        if rank == 0:
            print("[Fine] Coarse weights loaded successfully into fine model.", flush=True)
        return True
    except Exception as e:
        if rank == 0:
            print(f"[Fine] load_state_dict(strict=True) failed when loading coarse weights: {e}", flush=True)
        return False


def train_ddp(
    rank: int,
    world_size: int,
    config_path: str,
    penetration_loss_fn: Optional[
        Callable[[List[torch.Tensor], Dict[str, Any]], torch.Tensor]
    ] = None,
    require_coarse_load: bool = False,
):
    cfg = load_model_config(config_path)
    base_seed = int(cfg.get("seed", 0))
    run_seed = base_seed + rank
    random.seed(run_seed)
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    torch.cuda.manual_seed_all(run_seed)

    log_file = cfg.get("log_file", "out/multi_sphere_flat/logs/fine_train.log")
    setup_logger(rank, log_file)

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    log_stage(rank, f"[Fine] starting cache preparation before process group setup, device={device}")

    if rank == 0:
        print(f"[Fine] Using DDP with {world_size} GPUs", flush=True)
        print(f"[Fine] Loading dataset according to {config_path} ...", flush=True)

    train_npz = cfg.get("train_npz", "data/multi_sphere_flat/preprocessed/fine/train.store")
    validate_npz = cfg.get("validate_npz", cfg.get("val_npz", "data/multi_sphere_flat/preprocessed/fine/validate.store"))
    knn_k = int(cfg.get("knn_k", 32))
    num_anchors = int(cfg.get("num_anchors", knn_k))
    edge_knn_k = int(cfg.get("edge_knn_k", knn_k))
    if edge_knn_k <= 0:
        raise ValueError(f"edge_knn_k must be positive, got {edge_knn_k}")
    cache_knn_k = max(knn_k, edge_knn_k)
    knn_chunk = int(cfg.get("knn_chunk", 1024))
    local_query_chunk = int(cfg.get("local_query_chunk", 256))
    cross_pair_chunk = int(cfg.get("cross_pair_chunk", 8))
    idx_cache_dir_train = cfg.get("idx_cache_dir_train", "data/multi_sphere_flat/idx_cache/fine_train")
    idx_cache_dir_validate = cfg.get(
        "idx_cache_dir_validate",
        cfg.get("idx_cache_dir_val", "data/multi_sphere_flat/idx_cache/fine_validate"),
    )
    use_knn_cache = bool(cfg.get("use_knn_cache", True))
    knn_cache_format = str(cfg.get("knn_cache_format", "aggregate"))
    dataset = MeshDatasetCached(
        npz_path=train_npz,
        cache_dir=idx_cache_dir_train,
        k=cache_knn_k,
        chunk=knn_chunk,
        use_knn_cache=use_knn_cache,
        knn_cache_format=knn_cache_format,
    )
    log_stage(rank, f"[Fine] dataset loaded, samples={len(dataset)}")

    if use_knn_cache:
        dataset.require_complete_cache()
        log_stage(rank, f"[Fine] Packaged KNN cache validated: {dataset.knn_cache_path}")

    log_stage(rank, "[Fine] starting process group setup")
    setup(rank, world_size)
    log_stage(rank, f"[Fine] process group ready, device={device}")

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=base_seed,
    )

    batch_size = int(cfg.get("batch_size", 1))
    grad_accum_steps = max(1, int(cfg.get("grad_accum_steps", 1)))
    num_workers = int(cfg.get("num_workers", 8))

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        pin_memory=True,
        drop_last=True,
    )

    d_model = int(cfg.get("d_model", 768))
    nhead = int(cfg.get("nhead", 6))
    pre_layers = int(cfg.get("pre_layers", 4))
    post_layers = int(cfg.get("post_layers", 4))
    ff_mult = int(cfg.get("ff_mult", 4))
    dropout = float(cfg.get("dropout", 0.1))
    droppath_rate = float(cfg.get("droppath_rate", 0.10))
    gamma_init = float(cfg.get("gamma_init", 1e-3))
    zero_init_residual = bool(cfg.get("zero_init_residual", False))
    zero_init_decoder = bool(cfg.get("zero_init_decoder", True))
    torch_compile_enabled = bool(cfg.get("torch_compile", True))
    torch_compile_backend = str(cfg.get("torch_compile_backend", "inductor"))
    torch_compile_mode = str(cfg.get("torch_compile_mode", "default"))
    torch_compile_fullgraph = bool(cfg.get("torch_compile_fullgraph", False))
    torch_compile_dynamic = bool(cfg.get("torch_compile_dynamic", False))
    torch_compile_suppress_errors = bool(
        cfg.get("torch_compile_suppress_errors", True)
    )

    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(False)
    verify_flash_attention_support(device, head_dim=d_model // nhead, key_length=knn_k)
    verify_flash_attention_support(device, head_dim=d_model // nhead, key_length=num_anchors)
    log_stage(rank, "[Fine] attention backend=FlashAttention (forced on CUDA; no SDPA fallback)")

    log_stage(rank, "[Fine] building model and moving it to CUDA")
    eager_model = Predictor(
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
    log_stage(rank, "[Fine] model is on CUDA")

    log_stage(rank, "[Fine] loading coarse weights if configured")
    coarse_loaded = _load_coarse_weights_if_available(eager_model, device, cfg, rank)
    if require_coarse_load and not coarse_loaded:
        raise RuntimeError(
            "This fine-stage experiment requires successful strict loading of "
            "the configured coarse checkpoint."
        )
    log_stage(rank, "[Fine] coarse-weight loading step finished")

    log_stage(rank, "[Fine] wrapping model with DDP")
    ddp_model = DDP(eager_model, device_ids=[rank])
    log_stage(rank, "[Fine] DDP wrapper ready")

    if torch_compile_enabled:
        log_stage(
            rank,
            "[Fine] enabling torch.compile after DDP "
            f"(backend={torch_compile_backend}, mode={torch_compile_mode}, "
            f"fullgraph={torch_compile_fullgraph}, dynamic={torch_compile_dynamic})",
        )
    model = compile_training_model(
        ddp_model,
        enabled=torch_compile_enabled,
        backend=torch_compile_backend,
        mode=torch_compile_mode,
        fullgraph=torch_compile_fullgraph,
        dynamic=torch_compile_dynamic,
        suppress_errors=torch_compile_suppress_errors,
    )

    lr = float(cfg.get("lr", 1.6e-4))
    weight_decay = float(cfg.get("weight_decay", 1e-4))
    min_lr = float(cfg.get("min_lr", 1e-5))
    total_epochs = int(cfg.get("epochs", 200))
    warmup_ratio = float(cfg.get("warmup_ratio", 0.04))
    center_loss_weight = float(cfg.get("center_loss_weight", 0.1))
    edge_loss_weight = float(cfg.get("edge_loss_weight", cfg.get("shape_loss_weight", 100.0)))
    step_profiler_enabled = bool(cfg.get("step_profiler_enabled", True))
    step_profiler_epoch = int(cfg.get("step_profiler_epoch", 1))
    step_profiler_warmup_steps = int(cfg.get("step_profiler_warmup_steps", 5))
    step_profiler_num_steps = int(cfg.get("step_profiler_num_steps", 20))
    c4_rotation_augmentation = bool(cfg.get("c4_rotation_augmentation", False))

    save_dir = cfg.get("save_dir", "out/multi_sphere_flat/checkpoint/fine")
    plot_file = cfg.get("plot_file", "out/multi_sphere_flat/plots/fine_training_metrics.png")
    os.makedirs(save_dir, exist_ok=True)

    if rank == 0:
        import json
        print(f"==== Raw config from {config_path} ====", flush=True)
        print(json.dumps(cfg, indent=2, ensure_ascii=False), flush=True)

        print("==== Effective hyper-parameters (fine) ====", flush=True)
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
            "cache_knn_k": cache_knn_k,
            "knn_chunk": knn_chunk,
            "local_query_chunk": local_query_chunk,
            "cross_pair_chunk": cross_pair_chunk,
            "droppath_rate": droppath_rate,
            "gamma_init": gamma_init,
            "zero_init_residual": zero_init_residual,
            "zero_init_decoder": zero_init_decoder,
            "torch_compile": torch_compile_enabled,
            "torch_compile_backend": torch_compile_backend,
            "torch_compile_mode": torch_compile_mode,
            "torch_compile_fullgraph": torch_compile_fullgraph,
            "torch_compile_dynamic": torch_compile_dynamic,
            "torch_compile_suppress_errors": torch_compile_suppress_errors,
            "lr": lr,
            "weight_decay": weight_decay,
            "min_lr": min_lr,
            "epochs": total_epochs,
            "warmup_ratio": warmup_ratio,
            "center_loss_weight": center_loss_weight,
            "edge_loss_weight": edge_loss_weight,
            "penetration_loss_fn": (
                penetration_loss_fn.__name__
                if penetration_loss_fn is not None
                else None
            ),
            "require_coarse_load": require_coarse_load,
            "step_profiler_enabled": step_profiler_enabled,
            "step_profiler_epoch": step_profiler_epoch,
            "step_profiler_warmup_steps": step_profiler_warmup_steps,
            "step_profiler_num_steps": step_profiler_num_steps,
            "c4_rotation_augmentation": c4_rotation_augmentation,
            "batch_size_per_gpu": batch_size,
            "grad_accum_steps": grad_accum_steps,
            "effective_batch_size": batch_size * world_size * grad_accum_steps,
            "train_npz": train_npz,
            "validate_npz": validate_npz,
            "idx_cache_dir_train": idx_cache_dir_train,
            "idx_cache_dir_validate": idx_cache_dir_validate,
            "knn_cache_format": knn_cache_format,
            "save_dir": save_dir,
            "log_file": log_file,
            "plot_file": plot_file,
        }, indent=2, ensure_ascii=False), flush=True)

    # Fused AdamW consumes GradScaler's found_inf tensor on device. The regular
    # AdamW path performs a blocking host scalar read on every update.
    optimizer = torch.optim.AdamW(
        eager_model.parameters(), lr=lr, weight_decay=weight_decay, fused=True
    )
    scaler = GradScaler(device="cuda")

    steps_per_epoch = len(dataloader)  # number of micro-batches per epoch on each rank
    update_steps_per_epoch = math.ceil(steps_per_epoch / grad_accum_steps)
    warmup_steps = warmup_ratio * total_epochs * update_steps_per_epoch
    total_steps = total_epochs * update_steps_per_epoch

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr=min_lr,
    )

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    best_validate_l1 = float("inf")
    step_profiler = StepProfiler(
        enabled=step_profiler_enabled,
        device=device,
        rank=rank,
        profile_epoch=step_profiler_epoch,
        warmup_steps=step_profiler_warmup_steps,
        num_steps=step_profiler_num_steps,
    )
    validation_evaluator = (
        ValidationEvaluator(
            test_data_path=validate_npz,
            model_config_path=config_path,
            cache_dir=idx_cache_dir_validate,
            split_name="validate",
            device=device,
        )
        if rank == 0
        else None
    )
    # Rank 0 may spend time preparing the persistent validation cache/loader.
    # Keep every rank out of the first DDP forward until that one-time setup is done.
    if dist.is_initialized():
        dist.barrier()

    for epoch in range(total_epochs):
        model.train()
        sampler.set_epoch(epoch)

        # Detached GPU accumulation avoids per-metric scalar device
        # synchronizations on every micro-step. Reduce once at epoch end.
        metric_count = 5 if penetration_loss_fn is not None else 4
        metric_totals = torch.zeros(metric_count, device=device, dtype=torch.float64)

        iters = len(dataloader)
        if rank == 0:
            log_stage(rank, f"[Fine] epoch {epoch+1}/{total_epochs} start, iters={iters}")
        pbar = tqdm(dataloader, desc=f"[Fine Epoch {epoch+1}]", disable=True)
        data_iter = iter(pbar)

        optimizer.zero_grad(set_to_none=True)
        cur_accum_steps = grad_accum_steps

        for batch_idx in range(iters):
            data_wait_start = time.perf_counter()
            batch = next(data_iter)
            data_wait_ms = (time.perf_counter() - data_wait_start) * 1000.0
            if rank == 0 and batch_idx == 0:
                log_stage(rank, f"[Fine] epoch {epoch+1} first batch received")
            # Use the actual number of micro-batches in the current accumulation group.
            # This keeps the last partial group correctly normalized when iters % grad_accum_steps != 0.
            if batch_idx % grad_accum_steps == 0:
                cur_accum_steps = min(grad_accum_steps, iters - batch_idx)
            is_update_step = ((batch_idx + 1) % grad_accum_steps == 0) or ((batch_idx + 1) == iters)
            step_profiler.begin_step(
                epoch=epoch,
                batch_idx=batch_idx,
                data_wait_ms=data_wait_ms,
                is_update_step=is_update_step,
            )
            (
                X0_list,        # list of T(B, V_i, 3)
                v0_list,        # list of T(B, 3)
                f0_list,        # list of T(B, 3)
                rho0_list,      # list of T(B, 1)
                fric_list,      # list of T(B, 3)
                ft, fspin, froll,   # T(B, 1)
                gz,                 # T(B, 1)
                Y_list,         # list of T(B, V_i, 3)
                idx_local_list, # list of T(B, V_i, k)
            ) = batch

            step_profiler.start("h2d")
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
            if c4_rotation_augmentation:
                X0_list, v0_list, f0_list, Y_list = apply_random_c4_rotation(
                    X0_list, v0_list, f0_list, Y_list
                )
            step_profiler.end("h2d")

            # DDP decides whether to synchronize gradients during forward, so
            # no_sync must cover both forward and backward for accumulation
            # micro-batches. The final micro-batch exits this context and
            # synchronizes the entire accumulated gradient once.
            sync_context = (
                ddp_model.no_sync()
                if not is_update_step
                else contextlib.nullcontext()
            )
            with sync_context:
                step_profiler.start("forward")
                with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
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
                    step_profiler.end("forward")
                    step_profiler.start("loss")

                    l1_sum = X_hat_list[0].new_zeros(())
                    denom = 0

                    for X_hat, Y in zip(X_hat_list, Y_list):
                        diff = (X_hat - Y).float()  # fp32 accumulate
                        l1_sum = l1_sum + diff.abs().sum()
                        denom += diff.numel()

                    l1 = l1_sum / max(denom, 1)

                    edge = knn_edge_loss_multi(
                        Y_list, X_hat_list, idx_local_for_edge,
                        mode="l1"
                    )
                    cen = center_loss_multi(X_hat_list, Y_list, mode="euclidean_sq")

                    loss = l1 + center_loss_weight * cen + edge_loss_weight * edge
                    penetration_loss = None
                    if penetration_loss_fn is not None:
                        penetration_loss = penetration_loss_fn(X_hat_list, cfg)
                        if penetration_loss.ndim != 0:
                            raise ValueError(
                                "penetration_loss_fn must return a scalar tensor, "
                                f"got shape {tuple(penetration_loss.shape)}"
                            )
                        loss = loss + penetration_loss
                    step_profiler.end("loss")

                loss_for_backward = loss / float(cur_accum_steps)

                step_profiler.start("backward")
                scaler.scale(loss_for_backward).backward()
                step_profiler.end("backward")

            step_profiler.start("optimizer")
            if is_update_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(eager_model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                optimizer.zero_grad(set_to_none=True)
            step_profiler.end("optimizer")

            metric_values = [loss.detach(), l1.detach(), edge.detach(), cen.detach()]
            if penetration_loss is not None:
                metric_values.append(penetration_loss.detach())
            metric_totals.add_(torch.stack(metric_values).to(dtype=torch.float64))

            profile_summary = step_profiler.finish_step()
            if profile_summary is not None:
                profile_text = StepProfiler.format_summary(profile_summary)
                print(profile_text, flush=True)
                for line in profile_text.splitlines():
                    logging.info(line)

        profile_summary = step_profiler.finish_epoch(epoch)
        if profile_summary is not None:
            profile_text = StepProfiler.format_summary(profile_summary)
            print(profile_text, flush=True)
            for line in profile_text.splitlines():
                logging.info(line)

        dist.all_reduce(metric_totals, op=dist.ReduceOp.SUM)
        metric_totals.div_(float(iters * world_size))
        avg_metrics = metric_totals.cpu().tolist()
        avg_loss, avg_l1, avg_edge, avg_cen = avg_metrics[:4]
        avg_penetration_loss = avg_metrics[4] if penetration_loss_fn is not None else None
        current_lr = scheduler.get_last_lr()[0]

        if rank == 0:
            train_parts = [
                f"loss={avg_loss:.8f}",
                f"l1={avg_l1:.8f}",
                f"edge={avg_edge:.8f}",
                f"cen={avg_cen:.8f}",
            ]
            if avg_penetration_loss is not None:
                train_parts.append(f"pen={avg_penetration_loss:.8f}")
            train_parts.append(f"lr={current_lr:.8f}")
            logging.info(f"Epoch {epoch+1}: train " + ", ".join(train_parts))

        if rank == 0:
            assert validation_evaluator is not None
            eval_l1_raw = validation_evaluator.run(eager_model, ckpt_epoch=epoch)

            eval_l1 = _coerce_eval_error(eval_l1_raw)
            eval_edge = _coerce_metric(eval_l1_raw, "eval_edge")
            eval_cen = _coerce_metric(eval_l1_raw, "eval_cen")
            eval_samples = _coerce_metric(eval_l1_raw, "num_samples")

            if eval_l1 is not None:
                eval_parts = []
                if eval_edge is not None and eval_cen is not None:
                    eval_error = eval_l1 + center_loss_weight * eval_cen + edge_loss_weight * eval_edge
                    eval_parts.append(f"error={eval_error:.8f}")
                eval_parts.append(f"l1={eval_l1:.8f}")
                if eval_edge is not None:
                    eval_parts.append(f"edge={eval_edge:.8f}")
                if eval_cen is not None:
                    eval_parts.append(f"cen={eval_cen:.8f}")
                if eval_samples is not None:
                    eval_parts.append(f"samples={int(eval_samples)}")
                logging.info(f"Epoch {epoch+1}: validate " + ", ".join(eval_parts))
                update_training_plot(log_file, plot_file)

            if eval_l1 is not None and eval_l1 < best_validate_l1:
                best_validate_l1 = eval_l1
                to_save = eager_model.state_dict()
                ckpt_obj = {
                    "model": to_save,
                    "epoch": epoch,
                    "best_validate_l1": best_validate_l1,
                    "eval_split": "validate",
                    "eval_metrics": eval_l1_raw,
                    "config": cfg,
                }
                torch.save(ckpt_obj, os.path.join(save_dir, "best_validate_error_model.pth"))

        if dist.is_initialized():
            dist.barrier()

    if rank == 0:
        print("[Fine] Training complete.", flush=True)

    cleanup()


if __name__ == "__main__":
    os.chdir(PROJECT_DIR)
    parser = argparse.ArgumentParser(description="Train fine model for a selected scene.")
    parser.add_argument("--scene", choices=sorted(SCENE_CONFIGS), default="flat")
    parser.add_argument("--config", type=str, default=None, help="Optional explicit model config path.")
    args = parser.parse_args()

    config_path = resolve_config_path(args.scene, args.config)
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No CUDA device found.")
    mp.spawn(train_ddp, args=(world_size, config_path), nprocs=world_size, join=True)
