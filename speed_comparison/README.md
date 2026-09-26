# PIM versus MuJoCo Speed Comparison

`benchmark.py` compares PIM terminal-mesh inference with a headless MuJoCo rollout on
the same held-out scene IDs. It supports Flat, Bumpy, Jenga, and Deformable.

The packaged Jenga MuJoCo XML scenes, assets, and recorded results remain available,
but the PIM-backed Jenga benchmark cannot be rerun because the archive omits the Jenga
PIM test store and KNN cache to meet the size constraint described in the root README.

## Included files

- `benchmark.py`: benchmark and JSON-report entry point.
- `assets/<scene>/`: only the static MuJoCo mesh and height-field files referenced by
  the packaged test XML files.
- `data/<scene>/scene/test_scene/`: complete held-out MuJoCo XML scenes.
- `results_eager_amp_run1/`: recorded Flat, Bumpy, Jenga, and short Deformable results.
- `results_eager_amp_deformable4s/`: the recorded Deformable benchmark report.

PIM weights, test stores, and KNN caches are read from the sibling `pim/` directory.
The fixed timing horizon and stopping settings are declared directly in `benchmark.py`.
This package includes the inputs and settings required for the documented benchmark, but does not include the original dataset-generation pipeline.

## Measurement definition

The model timer measures only the synchronized PIM forward pass after checkpoint,
dataset, KNN cache, and device inputs are ready. MuJoCo reports both:

- steady-state time: state setup, rollout, and endpoint extraction; and
- end-to-end time: XML read/compile plus all steady-state work.

Single-scene latency compares PIM batch size 1 with one MuJoCo worker. Throughput pairs
model batch size `B` with `B` independent MuJoCo workers. XML compilation and process
startup are never silently mixed into the steady-state comparison.

## Run a smoke benchmark

Use an otherwise idle CPU and GPU. From the archive root:

```bash
conda activate pim
cd PIM_supplementary_code

python speed_comparison/benchmark.py \
  --scene flat \
  --backend all \
  --num-scenes 8 \
  --mujoco-warmup-scenes 1 \
  --mujoco-worker-counts 1,2,4,8 \
  --model-batch-sizes 1,2,4,8 \
  --model-warmup 5 \
  --model-repeats 10 \
  --results-dir speed_comparison/results_smoke
```

The command runs no viewer and creates four JSON files under
`speed_comparison/results_smoke/flat/`: `selection.json`, `mujoco.json`, `model.json`,
and `comparison.json`.

## Full protocol

For Flat, Bumpy, and Jenga, use 100 measured scenes, three warm-up scenes, model batch
sizes and MuJoCo worker counts `1,2,4,8`, 20 model warm-up calls, and 100 timed calls.
Replace `<scene>` below with `flat`, `bumpy`, or `jenga`:

```bash
python speed_comparison/benchmark.py \
  --scene <scene> --backend all --num-scenes 100 --seed 0 \
  --mujoco-warmup-scenes 3 --mujoco-worker-counts 1,2,4,8 \
  --model-batch-sizes 1,2,4,8 --model-warmup 20 --model-repeats 100 \
  --results-dir speed_comparison/results_reproduction
```

The submission's short Deformable timing uses ten measured scenes, one warm-up scene,
a 4 s benchmark-only horizon, and velocity threshold `1e-2`:

```bash
python speed_comparison/benchmark.py \
  --scene deformable --backend all --num-scenes 10 --seed 0 \
  --mujoco-warmup-scenes 1 --mujoco-worker-counts 1,2,4,8 \
  --model-batch-sizes 1,2,4,8 --deformable-sim-time 4.0 \
  --deformable-v-thresh 1e-2 --model-warmup 20 --model-repeats 100 \
  --results-dir speed_comparison/results_reproduction
```

That Deformable override is for timing only; it must not be used as an accuracy
protocol. Runtime results depend on the GPU, CPU, driver, process contention, power
state, and software stack, so record those details with every rerun.
