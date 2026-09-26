# PIM Supplementary Code

Given the size of the full datasets and the breadth of the complete research repository, this supplementary archive focuses on the artifacts required to inspect and reproduce the results documented below. It includes the relevant code, trained weights, held-out test data, evaluation entry points, and reference outputs. Large training datasets and components not required by the documented evaluation workflows are outside the scope of this archive and are identified explicitly in the corresponding sections.

 The documentation and directory structure in this
archive were prepared with generative AI assistance and reviewed by the authors. PIM
predicts complete terminal meshes directly from the initial scene; it is not an
autoregressive trajectory model.

## Archive contents

| Directory | Contents |
|---|---|
| `pim/` | PIM model, four scene-specific coarse/fine training configurations, fine checkpoints, evaluator, and reference metrics. Held-out test stores and KNN caches are included for Flat, Bumpy, and Deformable; the Jenga evaluation data are omitted as explained below. |
| `direct_baselines/` | Model-only releases of the two direct terminal-state baselines, renamed to **Rigid-Pose MLP** and **Vertex Transformer**, with paper-matched constructor defaults. |
| `rollout_baselines/` | A runnable controlled-Flat PIM evaluation with PIM test data, checkpoint, predictions, and metrics; plus model code only for the six autoregressive baselines. |
| `speed_comparison/` | The PIM-versus-MuJoCo benchmark, MuJoCo test XML files/assets, and the recorded benchmark outputs used by the paper. |
| `environments/` | Miniconda setup instructions for the `pim` and `hopnet` environments, reproducible environment files, and a lightweight installation check. |

Every top-level section has its own `README.md` describing the files and exact usage.

## Repository size and omitted Jenga data

Before the Jenga PIM evaluation dataset was removed, this packaged directory occupied
2,331,114,303 bytes (2.171 GiB). The Jenga held-out fine test store and its deterministic
KNN cache occupied 374,506,906 bytes (357.158 MiB): 98.776 MiB for the preprocessed test
store and 258.382 MiB for the cache. They are omitted to keep the repository below the
2 GiB storage quota presented by the public Anonymous GitHub service. After this
omission, the package occupies approximately 1.822 GiB.

The Jenga model configuration, fine checkpoint, reference metric, MuJoCo XML scenes and
assets, and recorded timing outputs remain available for inspection. However, the Jenga
PIM evaluation and the PIM-backed portion of the Jenga timing benchmark cannot be rerun
from this package because their PIM test store and KNN cache are not included.

## Recommended order

1. Follow `environments/README.md` and create both Conda environments.
2. Run the lightweight environment checks.
3. Reproduce the Flat, Bumpy, and Deformable PIM evaluations from `pim/`; inspect the
   packaged Jenga configuration, checkpoint, and reference metric separately.
4. Reproduce the controlled Flat PIM result from `rollout_baselines/`; inspect the
   code-only autoregressive baseline implementations separately.
5. Run the timing benchmark from `speed_comparison/` only on an otherwise idle machine.

All commands in this archive assume Linux, an NVIDIA GPU, and execution from the
directory explicitly stated in the relevant README. Paths are relative; the archive
may be extracted anywhere.

## Reproducibility scope

- The full held-out PIM test sets and their deterministic KNN caches are included for
  Flat, Bumpy, and Deformable. The Jenga test store and cache are omitted for the size
  reason documented above.
- Fine PIM checkpoints are included because they are the checkpoints used for the
  reported inference results. Coarse checkpoints and PIM training data are not needed
  for evaluation and are not included.
- The controlled Flat PIM test store, KNN cache, checkpoint, predictions, metrics, and
  evaluator are included under `rollout_baselines/pim_evaluation/`.
- The autoregressive baseline directories provide model implementations for reference; runnable training and evaluation artifacts are not included in this archive.
- No dataset-generation or preprocessing program is included. Dataset stores and KNN
  caches in the archive are completed, read-only evaluation inputs.
- Training configurations and training code are provided, but the large training and
  validation datasets are not duplicated in this archive.
- Runtime values are hardware- and software-dependent. The included JSON files record
  the measurements used by the submission; reruns should report their own hardware.

## Important execution note

The PIM evaluations are GPU workloads. The code-only rollout baseline section is not a
runnable reproduction package and has no evaluation commands. No renderer
or GUI is required by the documented PIM evaluation commands.
