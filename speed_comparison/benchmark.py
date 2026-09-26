#!/usr/bin/env python3
"""Benchmark endpoint prediction speed against MuJoCo test-scene rollouts."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import platform
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

import numpy as np
from tqdm import tqdm


PROJECT_DIR = Path(__file__).resolve().parents[1]
PIM_DIR = PROJECT_DIR / "pim"
SPEED_DIR = PROJECT_DIR / "speed_comparison"
if str(PIM_DIR) not in sys.path:
    sys.path.insert(0, str(PIM_DIR))

from utils.array_store import load_dataset_arrays


SCENES = {
    "flat": {
        "model_config": PIM_DIR / "model/multi_sphere_flat/model_config_fine.json",
        "scene_dir": SPEED_DIR / "data/multi_sphere_flat/scene/test_scene",
        "sim_config": {
            "sim_time": 10.0,
            "v_thresh": 0.001,
            "hold_time": 0.2,
            "force_duration": 0.05,
        },
    },
    "bumpy": {
        "model_config": PIM_DIR / "model/multi_sphere_bumpy/model_config_fine.json",
        "scene_dir": SPEED_DIR / "data/multi_sphere_bumpy/scene/test_scene",
        "sim_config": {
            "sim_time": 10.0,
            "v_thresh": 0.001,
            "hold_time": 0.2,
            "force_duration": 0.05,
        },
    },
    "jenga": {
        "model_config": PIM_DIR / "model/jenga_tower/model_config_fine.json",
        "scene_dir": SPEED_DIR / "data/jenga_tower/scene/test_scene",
        "sim_config": {
            "sim_time": 10.0,
            "v_thresh": 0.001,
            "hold_time": 0.2,
            "force_duration": 0.05,
        },
    },
    "deformable": {
        "model_config": PIM_DIR / "model/deformable_box/model_config_fine.json",
        "scene_dir": SPEED_DIR / "data/deformable_box/scene/test_scene",
        "sim_config": {
            "sim_time": 8.0,
            "v_thresh": 0.001,
            "hold_time": 0.2,
            "force_duration": 0.05,
        },
    },
}

FORCE_PATTERN = re.compile(
    r"init_force_([A-Za-z0-9_]+)\s+"
    r"([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)"
)
SCENE_INDEX_PATTERN = re.compile(r"(\d+)$")

_MUJOCO_WORKER_MODULE: Any = None


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_DIR / candidate


def resolve_pim_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PIM_DIR / candidate


def summarize_ms(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot summarize an empty timing list")
    return {
        "mean_ms": float(array.mean()),
        "std_ms": float(array.std()),
        "median_ms": float(np.median(array)),
        "p95_ms": float(np.percentile(array, 95)),
        "min_ms": float(array.min()),
        "max_ms": float(array.max()),
    }


def load_scene_ids(model_cfg: dict[str, Any]) -> tuple[list[str], Path]:
    test_path = resolve_pim_path(str(model_cfg["test_npz"]))
    arrays, _, resolved_path = load_dataset_arrays(test_path)
    if "scene_ids" not in arrays:
        raise ValueError(f"{resolved_path} is missing scene_ids")
    scene_ids = [str(value) for value in np.asarray(arrays["scene_ids"]).tolist()]
    if not scene_ids:
        raise ValueError(f"{resolved_path} contains no test scenes")
    return scene_ids, resolved_path


def select_indices(
    total: int,
    num_scenes: int,
    warmup_scenes: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    if num_scenes <= 0 or warmup_scenes < 0:
        raise ValueError("num_scenes must be positive and warmup_scenes non-negative")
    requested = num_scenes + warmup_scenes
    if requested > total:
        raise ValueError(
            f"Requested {requested} measured+warmup scenes, but the test split has {total}"
        )
    permutation = np.random.default_rng(seed).permutation(total)
    warmup = [int(i) for i in permutation[:warmup_scenes]]
    measured = [int(i) for i in permutation[warmup_scenes:requested]]
    return warmup, measured


def scene_xml_path(scene_dir: Path, scene_id: str) -> Path:
    match = SCENE_INDEX_PATTERN.search(scene_id)
    if match is None:
        raise ValueError(f"Cannot extract a numeric scene index from {scene_id!r}")
    path = scene_dir / f"scene_{int(match.group(1))}.xml"
    if not path.is_file():
        raise FileNotFoundError(f"MuJoCo scene XML not found: {path}")
    return path


def parse_initial_forces(xml_text: str) -> dict[str, np.ndarray]:
    forces = {}
    for match in FORCE_PATTERN.finditer(xml_text):
        forces[match.group(1)] = np.asarray(
            [float(match.group(2)), float(match.group(3)), float(match.group(4))],
            dtype=np.float64,
        )
    if not forces:
        raise ValueError("Scene XML contains no init_force_<object> comments")
    return forces


def reset_mujoco_state(mujoco: Any, model: Any, data: Any) -> None:
    if int(model.nkey) > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)


def resolve_force_bodies(
    mujoco: Any,
    model: Any,
    forces: dict[str, np.ndarray],
) -> dict[int, np.ndarray]:
    force_bodies = {}
    for name, force in forces.items():
        body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"object_{name}"
        )
        if body_id < 0:
            if float(np.linalg.norm(force)) > 0.0:
                raise ValueError(f"Cannot resolve body object_{name} for non-zero force")
            continue
        force_bodies[int(body_id)] = force
    return force_bodies


def rollout_mujoco(
    mujoco: Any,
    model: Any,
    data: Any,
    *,
    force_bodies: dict[int, np.ndarray],
    sim_time: float,
    force_duration: float,
    v_thresh: float,
    hold_time: float,
) -> tuple[int, float]:
    dt = float(model.opt.timestep)
    need_hold = max(1, int(hold_time / dt))
    held = 0
    steps = 0
    start_time = float(data.time)
    target_time = start_time + sim_time

    while float(data.time) < target_time:
        if float(data.time) < force_duration:
            for body_id, force in force_bodies.items():
                data.xfrc_applied[body_id, 0:3] = force
        else:
            for body_id in force_bodies:
                data.xfrc_applied[body_id, :] = 0.0

        mujoco.mj_step(model, data)
        mujoco.mj_normalizeQuat(model, data.qpos)
        steps += 1

        vmax = 0.0
        if data.qvel.size:
            vmax = float(np.abs(np.asarray(data.qvel)).max())
        if vmax < v_thresh:
            held += 1
            if held >= need_hold:
                break
        else:
            held = 0

    return steps, float(data.time) - start_time


def extract_mujoco_endpoint(
    mujoco: Any,
    model: Any,
    data: Any,
    object_names: list[str],
) -> tuple[int, float]:
    total_vertices = 0
    checksum = 0.0
    for name in object_names:
        flex_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, name)
        if flex_id >= 0:
            start = int(model.flex_vertadr[flex_id])
            count = int(model.flex_vertnum[flex_id])
            vertices = np.asarray(
                data.flexvert_xpos[start:start + count], dtype=np.float32
            ).copy()
        else:
            mesh_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_MESH, f"mesh_{name}"
            )
            geom_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_col"
            )
            body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, f"object_{name}"
            )
            if mesh_id < 0 or (geom_id < 0 and body_id < 0):
                raise ValueError(f"Cannot resolve output geometry for object {name}")
            start = int(model.mesh_vertadr[mesh_id])
            count = int(model.mesh_vertnum[mesh_id])
            mesh_vertices = np.asarray(model.mesh_vert)
            if mesh_vertices.ndim == 1:
                local_vertices = mesh_vertices[
                    3 * start:3 * (start + count)
                ].astype(np.float32).reshape(count, 3)
            else:
                local_vertices = np.asarray(
                    mesh_vertices[start:start + count], dtype=np.float32
                )
            if geom_id >= 0:
                position = np.asarray(data.geom_xpos[geom_id], dtype=np.float32)
                rotation = np.asarray(data.geom_xmat[geom_id], dtype=np.float32).reshape(3, 3)
            else:
                position = np.asarray(data.xpos[body_id], dtype=np.float32)
                rotation = np.asarray(data.xmat[body_id], dtype=np.float32).reshape(3, 3)
            vertices = local_vertices @ rotation.T + position

        total_vertices += int(vertices.shape[0])
        checksum += float(vertices.sum(dtype=np.float64))
    return total_vertices, checksum


def benchmark_one_mujoco_scene(
    mujoco: Any,
    xml_path: Path,
    object_names: list[str],
    sim_cfg: dict[str, float],
) -> dict[str, Any]:
    end_to_end_start = time.perf_counter()
    xml_text = xml_path.read_text(encoding="utf-8")
    forces = parse_initial_forces(xml_text)

    compile_start = time.perf_counter()
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    compile_ms = (time.perf_counter() - compile_start) * 1000.0

    setup_start = time.perf_counter()
    data = mujoco.MjData(model)
    reset_mujoco_state(mujoco, model, data)
    force_bodies = resolve_force_bodies(mujoco, model, forces)
    setup_ms = (time.perf_counter() - setup_start) * 1000.0

    rollout_start = time.perf_counter()
    steps, simulated_seconds = rollout_mujoco(
        mujoco,
        model,
        data,
        force_bodies=force_bodies,
        sim_time=sim_cfg["sim_time"],
        force_duration=sim_cfg["force_duration"],
        v_thresh=sim_cfg["v_thresh"],
        hold_time=sim_cfg["hold_time"],
    )
    rollout_ms = (time.perf_counter() - rollout_start) * 1000.0

    output_start = time.perf_counter()
    total_vertices, output_checksum = extract_mujoco_endpoint(
        mujoco, model, data, object_names
    )
    output_ms = (time.perf_counter() - output_start) * 1000.0
    end_to_end_ms = (time.perf_counter() - end_to_end_start) * 1000.0

    return {
        "xml": str(xml_path.relative_to(PROJECT_DIR)),
        "compile_ms": compile_ms,
        "state_setup_ms": setup_ms,
        "rollout_ms": rollout_ms,
        "endpoint_extraction_ms": output_ms,
        "steady_state_ms": setup_ms + rollout_ms + output_ms,
        "end_to_end_ms": end_to_end_ms,
        "steps": steps,
        "simulated_seconds": simulated_seconds,
        "total_vertices": total_vertices,
        "output_checksum": output_checksum,
    }


def _initialize_mujoco_worker(ready_queue: Any = None) -> None:
    """Import MuJoCo once in each spawned worker before timing starts."""
    global _MUJOCO_WORKER_MODULE
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    import mujoco

    _MUJOCO_WORKER_MODULE = mujoco
    if ready_queue is not None:
        ready_queue.put(os.getpid())


def _mujoco_worker_ready(token: int) -> int:
    if _MUJOCO_WORKER_MODULE is None:
        _initialize_mujoco_worker()
    return int(token)


def _run_mujoco_worker_job(
    job: tuple[str, list[str], dict[str, float]],
) -> dict[str, Any]:
    if _MUJOCO_WORKER_MODULE is None:
        _initialize_mujoco_worker()
    xml_path, object_names, sim_cfg = job
    record = benchmark_one_mujoco_scene(
        _MUJOCO_WORKER_MODULE,
        Path(xml_path),
        object_names,
        sim_cfg,
    )
    record["worker_pid"] = os.getpid()
    return record


def estimate_parallel_steady_state_wall_ms(
    records: list[dict[str, Any]],
) -> float:
    """Estimate parallel steady-state wall time from per-worker busy time."""
    busy_ms_by_worker: dict[int, float] = {}
    for record in records:
        worker_pid = int(record["worker_pid"])
        busy_ms_by_worker[worker_pid] = (
            busy_ms_by_worker.get(worker_pid, 0.0)
            + float(record["steady_state_ms"])
        )
    return max(busy_ms_by_worker.values(), default=0.0)


def run_mujoco_jobs(
    jobs: list[tuple[str, list[str], dict[str, float]]],
    warmup_jobs: list[tuple[str, list[str], dict[str, float]]],
    worker_count: int,
    *,
    scene: str,
) -> tuple[list[dict[str, Any]], float, float]:
    """Run warmup jobs, then time measured jobs with a fixed worker count.

    Worker process startup and module import happen before the measured wall
    clock. Task dispatch and worker execution are included in parallel wall
    time, which is the quantity used for throughput comparison.
    """
    if worker_count <= 0:
        raise ValueError(f"worker_count must be positive, got {worker_count}")

    if worker_count == 1:
        for job in tqdm(
            warmup_jobs,
            desc=f"MuJoCo warmup {scene} ({worker_count} worker)",
            unit="scene",
        ):
            _run_mujoco_worker_job(job)
        records = []
        start = time.perf_counter()
        for job in tqdm(
            jobs,
            desc=f"MuJoCo benchmark {scene} ({worker_count} worker)",
            unit="scene",
        ):
            records.append(_run_mujoco_worker_job(job))
        wall_ms = (time.perf_counter() - start) * 1000.0
        return records, wall_ms, estimate_parallel_steady_state_wall_ms(records)

    context = multiprocessing.get_context("spawn")
    ready_queue = context.Queue()
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=context,
        initializer=_initialize_mujoco_worker,
        initargs=(ready_queue,),
    ) as executor:
        ready_futures = [
            executor.submit(_mujoco_worker_ready, token)
            for token in range(worker_count)
        ]
        # Each initializer signals exactly once. Waiting for all signals keeps
        # process startup and MuJoCo imports outside the measured wall clock.
        for _ in range(worker_count):
            ready_queue.get()
        for future in ready_futures:
            future.result()
        list(
            tqdm(
                executor.map(_run_mujoco_worker_job, warmup_jobs, chunksize=1),
                total=len(warmup_jobs),
                desc=f"MuJoCo warmup {scene} ({worker_count} workers)",
                unit="scene",
            )
        )
        start = time.perf_counter()
        records = list(
            tqdm(
                executor.map(_run_mujoco_worker_job, jobs, chunksize=1),
                total=len(jobs),
                desc=f"MuJoCo benchmark {scene} ({worker_count} workers)",
                unit="scene",
            )
        )
        wall_ms = (time.perf_counter() - start) * 1000.0
    return records, wall_ms, estimate_parallel_steady_state_wall_ms(records)


def benchmark_mujoco(
    scene: str,
    model_cfg: dict[str, Any],
    scene_ids: list[str],
    warmup_indices: list[int],
    measured_indices: list[int],
    worker_counts: list[int],
    deformable_sim_time: float,
    deformable_v_thresh: float,
) -> dict[str, Any]:
    import mujoco

    scene_spec = SCENES[scene]
    scene_dir = Path(scene_spec["scene_dir"])
    sim_cfg = {
        key: float(value)
        for key, value in dict(scene_spec["sim_config"]).items()
    }
    source_sim_cfg = dict(sim_cfg)
    if scene == "deformable":
        sim_cfg["sim_time"] = float(deformable_sim_time)
        sim_cfg["v_thresh"] = float(deformable_v_thresh)
    arrays, _, _ = load_dataset_arrays(resolve_pim_path(str(model_cfg["test_npz"])))
    object_names = [str(value) for value in np.asarray(arrays["obj_names"]).tolist()]

    warmup_jobs = [
        (
            str(scene_xml_path(scene_dir, scene_ids[index])),
            object_names,
            sim_cfg,
        )
        for index in warmup_indices
    ]
    measured_jobs = [
        (
            str(scene_xml_path(scene_dir, scene_ids[index])),
            object_names,
            sim_cfg,
        )
        for index in measured_indices
    ]

    timing_keys = (
        "compile_ms",
        "state_setup_ms",
        "rollout_ms",
        "endpoint_extraction_ms",
        "steady_state_ms",
        "end_to_end_ms",
    )
    worker_results: dict[str, dict[str, Any]] = {}
    for worker_count in worker_counts:
        records, parallel_wall_ms, parallel_steady_state_wall_ms = run_mujoco_jobs(
            measured_jobs,
            warmup_jobs,
            worker_count,
            scene=scene,
        )
        for index, record in zip(measured_indices, records):
            record["test_index"] = index
            record["scene_id"] = scene_ids[index]

        wall_seconds = parallel_wall_ms / 1000.0
        worker_results[str(worker_count)] = {
            "worker_count": worker_count,
            "num_warmup_scenes": len(warmup_indices),
            "num_measured_scenes": len(measured_indices),
            "process_startup_included": False,
            "parallel_wall_ms": parallel_wall_ms,
            "parallel_end_to_end_samples_per_second": (
                len(records) / max(wall_seconds, 1e-12)
            ),
            "parallel_steady_state_wall_ms_estimate": (
                parallel_steady_state_wall_ms
            ),
            "parallel_steady_state_samples_per_second_estimate": (
                len(records)
                / max(parallel_steady_state_wall_ms / 1000.0, 1e-12)
            ),
            "parallel_simulation_seconds_per_wall_second": float(
                sum(record["simulated_seconds"] for record in records)
                / max(wall_seconds, 1e-12)
            ),
            "timings": {
                key: summarize_ms([float(record[key]) for record in records])
                for key in timing_keys
            },
            "mean_steps": float(np.mean([record["steps"] for record in records])),
            "mean_simulated_seconds": float(
                np.mean([record["simulated_seconds"] for record in records])
            ),
            "simulation_seconds_per_rollout_wall_second": float(
                sum(record["simulated_seconds"] for record in records)
                / max(sum(record["rollout_ms"] for record in records) / 1000.0, 1e-12)
            ),
            "per_scene": records,
        }

    return {
        "backend": "mujoco",
        "scene": scene,
        "mujoco_version": getattr(mujoco, "__version__", "unknown"),
        "source_sim_config": source_sim_cfg,
        "sim_config": sim_cfg,
        "benchmark_only_override": (
            {"sim_time": deformable_sim_time, "v_thresh": deformable_v_thresh}
            if scene == "deformable"
            else None
        ),
        "object_names": object_names,
        "worker_counts": worker_counts,
        "workers": worker_results,
    }


def load_state_dict(checkpoint: dict[str, Any], torch: Any) -> dict[str, Any]:
    for key in ("model", "state_dict", "ema", "module", "net"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            state_dict = checkpoint[key]
            break
    else:
        if isinstance(checkpoint, dict) and all(
            isinstance(value, torch.Tensor) for value in checkpoint.values()
        ):
            state_dict = checkpoint
        else:
            raise RuntimeError("Checkpoint has no state-dict-like entry")
    return {
        key.removeprefix("module."): value for key, value in state_dict.items()
    }


def build_predictor(torch: Any, model_cfg: dict[str, Any], device: Any) -> tuple[Any, dict[str, Any]]:
    from model.transformer import Predictor, verify_flash_attention_support

    checkpoint_path = Path(model_cfg["save_dir"]) / "best_validate_error_model.pth"
    checkpoint_path = resolve_pim_path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_cfg = checkpoint.get("config", {})
    if not isinstance(checkpoint_cfg, dict):
        checkpoint_cfg = {}
    cfg = {**model_cfg, **checkpoint_cfg}
    kwargs = {
        "d_model": int(cfg.get("d_model", 512)),
        "nhead": int(cfg.get("nhead", 8)),
        "pre_layers": int(cfg.get("pre_layers", 4)),
        "post_layers": int(cfg.get("post_layers", 4)),
        "ff_mult": int(cfg.get("ff_mult", 4)),
        "dropout": float(cfg.get("dropout", 0.0)),
        "knn_k": int(cfg.get("knn_k", 32)),
        "knn_chunk": int(cfg.get("knn_chunk", 1024)),
        "local_query_chunk": int(cfg.get("local_query_chunk", 256)),
        "cross_pair_chunk": int(cfg.get("cross_pair_chunk", 8)),
        "droppath_rate": float(cfg.get("droppath_rate", 0.1)),
        "gamma_init": float(cfg.get("gamma_init", 1e-3)),
        "zero_init_residual": bool(cfg.get("zero_init_residual", False)),
        "zero_init_decoder": bool(cfg.get("zero_init_decoder", True)),
    }
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)
        verify_flash_attention_support(
            device,
            head_dim=kwargs["d_model"] // kwargs["nhead"],
            key_length=kwargs["knn_k"],
        )

    model = Predictor(**kwargs).to(device)
    model.load_state_dict(load_state_dict(checkpoint, torch), strict=True)
    model.eval()
    return model, {"checkpoint": str(checkpoint_path.relative_to(PROJECT_DIR)), **kwargs}


def prepare_model_inputs(dataset: Any, indices: list[int], device: Any) -> dict[str, Any]:
    from torch.utils.data import default_collate

    batch = default_collate([dataset[index] for index in indices])
    (
        x0_list,
        v0_list,
        f0_list,
        rho0_list,
        fric_list,
        ft,
        fspin,
        froll,
        gz,
        _y_list,
        idx_local_list,
    ) = batch
    return {
        "X0_list": [value.to(device) for value in x0_list],
        "v0_list": [value.to(device) for value in v0_list],
        "f0_list": [value.to(device) for value in f0_list],
        "rho0_list": [value.to(device) for value in rho0_list],
        "fric_list": [value.to(device) for value in fric_list],
        "ft": ft.to(device),
        "fspin": fspin.to(device),
        "froll": froll.to(device),
        "gz": gz.to(device),
        "idx_local_list": [value.to(device) for value in idx_local_list],
    }


def time_model_forward(
    torch: Any,
    model: Any,
    model_inputs: dict[str, Any],
    device: Any,
    warmup: int,
    repeats: int,
    amp: bool,
) -> list[float]:
    amp_context: Callable[[], Any]
    if device.type == "cuda":
        amp_context = lambda: torch.autocast(device_type="cuda", enabled=amp)
    else:
        amp_context = nullcontext

    with torch.inference_mode(), amp_context():
        for _ in range(warmup):
            model(**model_inputs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
        with torch.inference_mode(), amp_context():
            for start, end in zip(starts, ends):
                start.record()
                model(**model_inputs)
                end.record()
        torch.cuda.synchronize(device)
        return [float(start.elapsed_time(end)) for start, end in zip(starts, ends)]

    timings = []
    with torch.inference_mode(), amp_context():
        for _ in range(repeats):
            start = time.perf_counter()
            model(**model_inputs)
            timings.append((time.perf_counter() - start) * 1000.0)
    return timings


def benchmark_model(
    scene: str,
    model_cfg: dict[str, Any],
    measured_indices: list[int],
    *,
    batch_sizes: list[int],
    warmup: int,
    repeats: int,
    device_name: str,
    amp: bool,
    compile_model: bool,
) -> dict[str, Any]:
    import torch
    from utils.mesh_dataset_cached import MeshDatasetCached

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA model benchmark requested, but CUDA is unavailable")
    device = torch.device(device_name)
    setup_start = time.perf_counter()
    model, model_info = build_predictor(torch, model_cfg, device)
    if compile_model:
        model = torch.compile(
            model,
            backend=str(model_cfg.get("torch_compile_backend", "inductor")),
            mode=str(model_cfg.get("torch_compile_mode", "default")),
            fullgraph=bool(model_cfg.get("torch_compile_fullgraph", False)),
            dynamic=bool(model_cfg.get("torch_compile_dynamic", False)),
        )

    dataset = MeshDatasetCached(
        npz_path=str(resolve_pim_path(str(model_cfg["test_npz"]))),
        cache_dir=str(resolve_pim_path(str(model_cfg["idx_cache_dir_test"]))),
        k=int(model_cfg.get("knn_k", 32)),
        chunk=int(model_cfg.get("knn_chunk", 1024)),
        use_knn_cache=True,
        knn_cache_format=str(model_cfg.get("knn_cache_format", "aggregate")),
    )
    if dataset._knn_store is None or not dataset._knn_store.is_complete:
        raise RuntimeError(
            "The configured test KNN cache is incomplete. Build it before benchmarking."
        )
    setup_seconds = time.perf_counter() - setup_start

    batch_results = []
    for batch_size in batch_sizes:
        if batch_size > len(measured_indices):
            raise ValueError(
                f"Model batch size {batch_size} exceeds {len(measured_indices)} selected scenes"
            )
        inputs = prepare_model_inputs(dataset, measured_indices[:batch_size], device)
        timings = time_model_forward(
            torch,
            model,
            inputs,
            device,
            warmup=warmup,
            repeats=repeats,
            amp=amp,
        )
        summary = summarize_ms(timings)
        batch_results.append(
            {
                "batch_size": batch_size,
                "batch_latency": summary,
                "mean_per_sample_ms": summary["mean_ms"] / batch_size,
                "samples_per_second": batch_size * 1000.0 / summary["mean_ms"],
                "raw_batch_ms": timings,
            }
        )

    result = {
        "backend": "model",
        "scene": scene,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "amp": bool(amp and device.type == "cuda"),
        "torch_compile": compile_model,
        "warmup_iterations": warmup,
        "timed_iterations": repeats,
        "setup_seconds": setup_seconds,
        "model": model_info,
        "batches": batch_results,
    }
    return result


def build_comparison(
    scene: str,
    mujoco_result: dict[str, Any],
    model_result: dict[str, Any],
) -> dict[str, Any]:
    batches = []
    for batch in model_result["batches"]:
        batch_size = int(batch["batch_size"])
        model_per_sample_ms = float(batch["mean_per_sample_ms"])
        worker_result = mujoco_result["workers"].get(str(batch_size))
        comparison = {
            "batch_size": batch_size,
            "mujoco_worker_count": batch_size if worker_result is not None else None,
            "model_mean_per_sample_ms": model_per_sample_ms,
            "model_samples_per_second": float(batch["samples_per_second"]),
        }
        if worker_result is None:
            comparison["matching_mujoco_worker_available"] = False
        else:
            mujoco_steady_ms = float(
                worker_result["timings"]["steady_state_ms"]["mean_ms"]
            )
            mujoco_e2e_ms = float(
                worker_result["timings"]["end_to_end_ms"]["mean_ms"]
            )
            mujoco_parallel_steady_sps = float(
                worker_result["parallel_steady_state_samples_per_second_estimate"]
            )
            mujoco_parallel_e2e_sps = float(
                worker_result["parallel_end_to_end_samples_per_second"]
            )
            comparison.update(
                {
                    "matching_mujoco_worker_available": True,
                    "mujoco_mean_steady_state_per_scene_ms": mujoco_steady_ms,
                    "mujoco_mean_end_to_end_per_scene_ms": mujoco_e2e_ms,
                    "mujoco_parallel_wall_ms": float(
                        worker_result["parallel_wall_ms"]
                    ),
                    "mujoco_parallel_steady_state_wall_ms_estimate": float(
                        worker_result["parallel_steady_state_wall_ms_estimate"]
                    ),
                    "mujoco_parallel_steady_state_samples_per_second_estimate": (
                        mujoco_parallel_steady_sps
                    ),
                    "mujoco_parallel_end_to_end_samples_per_second": (
                        mujoco_parallel_e2e_sps
                    ),
                    "throughput_speedup_vs_mujoco_parallel_steady_state": (
                        float(batch["samples_per_second"])
                        / mujoco_parallel_steady_sps
                    ),
                    "throughput_speedup_vs_mujoco_parallel_end_to_end": (
                        float(batch["samples_per_second"])
                        / mujoco_parallel_e2e_sps
                    ),
                }
            )
            if batch_size == 1:
                comparison.update(
                    {
                        "latency_speedup_vs_mujoco_steady_state": (
                            mujoco_steady_ms / model_per_sample_ms
                        ),
                        "latency_speedup_vs_mujoco_end_to_end": (
                            mujoco_e2e_ms / model_per_sample_ms
                        ),
                    }
                )
        batches.append(comparison)
    return {
        "scene": scene,
        "primary_comparison": (
            "Matching model batch size with the same number of concurrent MuJoCo workers"
        ),
        "model_batches": batches,
    }


def write_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(f"Saved: {path}")


def parse_batch_sizes(value: str) -> list[int]:
    try:
        batch_sizes = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Batch sizes must be comma-separated integers") from exc
    if not batch_sizes or any(value <= 0 for value in batch_sizes):
        raise argparse.ArgumentTypeError("Batch sizes must be positive")
    if len(set(batch_sizes)) != len(batch_sizes):
        raise argparse.ArgumentTypeError("Batch sizes must be unique")
    return batch_sizes


def system_info() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "logical_cpu_count": os.cpu_count(),
    }


def run_scene(scene: str, args: argparse.Namespace) -> None:
    scene_spec = SCENES[scene]
    model_cfg = load_json(scene_spec["model_config"])
    scene_ids, test_path = load_scene_ids(model_cfg)
    num_scenes = (
        args.num_scenes
        if args.num_scenes is not None
        else (10 if scene == "deformable" else 100)
    )
    mujoco_warmup_scenes = (
        args.mujoco_warmup_scenes
        if args.mujoco_warmup_scenes is not None
        else (1 if scene == "deformable" else 3)
    )
    required_parallelism_values = []
    if args.backend in {"model", "all"}:
        required_parallelism_values.append(max(args.model_batch_sizes))
    if args.backend in {"mujoco", "all"}:
        required_parallelism_values.append(max(args.mujoco_worker_counts))
    required_parallelism = max(required_parallelism_values)
    if num_scenes < required_parallelism:
        raise ValueError(
            f"Scene {scene!r} needs --num-scenes >= {required_parallelism} for the "
            "requested model batches and MuJoCo workers"
        )
    warmup_indices, measured_indices = select_indices(
        len(scene_ids), num_scenes, mujoco_warmup_scenes, args.seed
    )
    output_dir = args.results_dir / scene
    selection = {
        "scene": scene,
        "seed": args.seed,
        "num_measured_scenes": num_scenes,
        "num_mujoco_warmup_scenes": mujoco_warmup_scenes,
        "test_data": str(test_path.relative_to(PROJECT_DIR)),
        "warmup_indices": warmup_indices,
        "measured_indices": measured_indices,
        "measured_scene_ids": [scene_ids[index] for index in measured_indices],
        "system": system_info(),
    }
    write_result(output_dir / "selection.json", selection)

    mujoco_result = None
    model_result = None
    if args.backend in {"mujoco", "all"}:
        mujoco_result = benchmark_mujoco(
            scene,
            model_cfg,
            scene_ids,
            warmup_indices,
            measured_indices,
            args.mujoco_worker_counts,
            args.deformable_sim_time,
            args.deformable_v_thresh,
        )
        mujoco_result["selection_seed"] = args.seed
        write_result(output_dir / "mujoco.json", mujoco_result)

    if args.backend in {"model", "all"}:
        model_result = benchmark_model(
            scene,
            model_cfg,
            measured_indices,
            batch_sizes=args.model_batch_sizes,
            warmup=args.model_warmup,
            repeats=args.model_repeats,
            device_name=args.device,
            amp=not args.no_amp,
            compile_model=args.torch_compile,
        )
        model_result["selection_seed"] = args.seed
        model_result["selected_test_indices"] = measured_indices
        write_result(output_dir / "model.json", model_result)

    if mujoco_result is not None and model_result is not None:
        comparison = build_comparison(scene, mujoco_result, model_result)
        write_result(output_dir / "comparison.json", comparison)
        for batch in comparison["model_batches"]:
            if not batch["matching_mujoco_worker_available"]:
                print(
                    f"[{scene}] batch={batch['batch_size']}: "
                    "no matching MuJoCo worker count; throughput comparison skipped"
                )
                continue
            message = (
                f"[{scene}] batch={batch['batch_size']} / workers={batch['mujoco_worker_count']}: "
                "throughput="
                f"{batch['throughput_speedup_vs_mujoco_parallel_steady_state']:.2f}x "
                "vs steady-state, "
                f"{batch['throughput_speedup_vs_mujoco_parallel_end_to_end']:.2f}x "
                "vs end-to-end"
            )
            if batch["batch_size"] == 1:
                message += (
                    f", latency={batch['latency_speedup_vs_mujoco_steady_state']:.2f}x"
                )
            print(message)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare endpoint model forward speed with MuJoCo rollout speed."
    )
    parser.add_argument("--scene", choices=[*SCENES, "all"], default="flat")
    parser.add_argument("--backend", choices=["mujoco", "model", "all"], default="all")
    parser.add_argument(
        "--num-scenes",
        type=int,
        default=None,
        help="Measured scenes. Defaults to 10 for deformable and 100 otherwise.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mujoco-warmup-scenes",
        type=int,
        default=None,
        help="Warmup scenes. Defaults to 1 for deformable and 3 otherwise.",
    )
    parser.add_argument(
        "--mujoco-worker-counts",
        type=parse_batch_sizes,
        default=[1, 2, 4, 8],
        help="Concurrent MuJoCo worker counts, comma-separated.",
    )
    parser.add_argument(
        "--deformable-sim-time",
        type=float,
        default=4.0,
        help="Benchmark-only deformable sim_time override (default: 4.0 s).",
    )
    parser.add_argument(
        "--deformable-v-thresh",
        type=float,
        default=1e-2,
        help="Benchmark-only deformable velocity threshold override (default: 1e-2).",
    )
    parser.add_argument(
        "--model-batch-sizes",
        type=parse_batch_sizes,
        default=[1, 2, 4, 8],
    )
    parser.add_argument("--model-warmup", type=int, default=10)
    parser.add_argument("--model-repeats", type=int, default=50)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=SPEED_DIR / "results",
    )
    args = parser.parse_args()
    if args.num_scenes is not None and args.num_scenes <= 0:
        parser.error("--num-scenes must be positive")
    if args.mujoco_warmup_scenes is not None and args.mujoco_warmup_scenes < 0:
        parser.error("--mujoco-warmup-scenes must be non-negative")
    if args.model_warmup < 0 or args.model_repeats <= 0:
        parser.error("model warmup must be non-negative and repeats positive")
    if args.deformable_sim_time <= 0.0:
        parser.error("--deformable-sim-time must be positive")
    if args.deformable_v_thresh <= 0.0:
        parser.error("--deformable-v-thresh must be positive")
    args.results_dir = resolve_project_path(args.results_dir)

    scenes = list(SCENES) if args.scene == "all" else [args.scene]
    for scene in scenes:
        run_scene(scene, args)


if __name__ == "__main__":
    main()
