# PIM Model and Four-Scene Evaluation

This directory contains the PIM implementation and artifacts for the four experiments
reported in the paper: Flat, Bumpy, Jenga, and Deformable. It is self-contained for
evaluating Flat, Bumpy, and Deformable. The Jenga test store and KNN cache are omitted
to satisfy the repository-size constraint documented in the archive-level README.

## Files

- `model/transformer.py`: the PIM architecture.
- `model/<scene>/model_config_coarse.json`: coarse-stage architecture and training
  parameters.
- `model/<scene>/model_config_fine.json`: fine-stage architecture, training parameters,
  test-data path, cache path, checkpoint path, and output path.
- `train_coarse.py` and `train_fine.py`: coarse-to-fine training entry points. Training
  data are not included in this supplementary archive.
- `evaluate.py`: shared four-scene evaluator.
- `utils/`: read-only array-store and KNN-cache loaders plus augmentation, compilation,
  evaluation, and training utilities used by the entry points.
- `data/<scene>/preprocessed/fine/test.store/`: complete held-out fine-resolution test
  split for Flat, Bumpy, and Deformable.
- `data/<scene>/idx_cache/fine_test/`: completed KNN cache corresponding to each included
  test store. The Jenga test store and cache are not packaged.
- `out/<scene>/checkpoint/fine/best_validate_error_model.pth`: validation-selected fine
  checkpoint used for the reported test result.
- `reference_results/<scene>/metrics.json`: metrics recorded from the packaged source
  experiment.

The internal scene directory names are retained for checkpoint and configuration
compatibility:

| Paper name | Internal directory | Test scenes |
|---|---|---:|
| Flat | `multi_sphere_flat` | 1,000 |
| Bumpy | `multi_sphere_bumpy` | 1,000 |
| Jenga | `jenga_tower` | 2,000 |
| Deformable | `deformable_box` | 200 |

## Evaluate the packaged checkpoints

From this directory:

```bash
conda activate pim
cd PIM_supplementary_code/pim

python evaluate.py --scene flat
python evaluate.py --scene bumpy
python evaluate.py --scene deformable
```

`python evaluate.py --scene jenga` requires the omitted Jenga test store and KNN cache
and therefore is not runnable from this package. Its configuration, checkpoint, and
recorded reference metric remain included for inspection.

Use `CUDA_VISIBLE_DEVICES=<gpu>` before a command to select a physical GPU. If memory is
limited, reduce `--batch-size` and optionally `--num-workers`; neither changes the model
or metric definitions. Each run writes `metrics.json`, `errors.txt`, predictions, and
per-scene vertex files under `out/<scene>/results/`.

The expected aggregate results are:

| Scene | Vertex L2 (m) | Centroid L2 (m) | KNN-edge error (m) |
|---|---:|---:|---:|
| Flat | 0.025694567 | 0.020983366 | 1.182380e-5 |
| Bumpy | 0.027547560 | 0.025535897 | 2.533382e-5 |
| Jenga | 0.020357886 | 0.019960931 | 8.217252e-5 |
| Deformable | 0.019304534 | 0.013210713 | 3.804902e-4 |

The authoritative full-precision values and sample counts are in `reference_results/`.
Minor last-digit changes can occur across GPU architectures or CUDA/PyTorch builds.

## Configuration notes

The JSON files preserve both architecture and optimizer settings used by the source
experiments. Fine inference loads the checkpoint configuration and then applies it to
the packaged model with strict state-dict matching. Fine-model training additionally
requires the corresponding coarse checkpoint and training/validation stores, which are
outside the scope of this evaluation package. The packaged test stores and KNN caches
provide the complete inputs required by the documented Flat, Bumpy, and Deformable
evaluation commands.
