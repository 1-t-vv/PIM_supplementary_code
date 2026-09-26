# Controlled-Flat PIM Evaluation

This directory reproduces
the PIM row of the controlled Flat rollout comparison on the same 100 held-out scenes.

## Files

- `test_data/test.store/`: PIM's preprocessed 100-scene test input and ground truth.
- `cache/fine_test/`: the checksum-guarded 32-neighbor cache for that exact store.
- `checkpoint/best_validate_error_model.pth`: the validation-selected PIM checkpoint.
- `model_config.json`: the model and evaluation parameters used by the checkpoint.
- `evaluate.py`: portable wrapper around the main PIM evaluator in `../../pim/`.
- `reference_results/`: the recorded PIM predictions, per-scene outputs, errors, and
  aggregate metrics. New evaluations are written elsewhere and do not overwrite them.

No dataset-generation or preprocessing code is included. The test store and KNN cache
are completed evaluation inputs.

## Evaluate

From the archive root:

```bash
conda activate pim
cd PIM_supplementary_code

python rollout_baselines/pim_evaluation/evaluate.py --gpus 0
```

Use `--batch-size` or `--num-workers` to fit the local machine. Results are written to
`rollout_baselines/pim_evaluation/reproduced_results/` by default. A path-only check
that does not initialize the model is available with:

```bash
python rollout_baselines/pim_evaluation/evaluate.py --dry-run
```

The packaged reference result contains 100 scenes and reports vertex L2 error
`0.030731391906738282`, centroid L2 error `0.027970449924468996`, and local KNN-edge
error `1.0872717248275875e-05`. Small last-digit differences may occur across CUDA and
GPU versions.
