# Linux Miniconda Environments

The supplementary code uses two isolated environments:

- `pim`: PIM evaluation, MuJoCo simulation, and PIM-versus-MuJoCo timing. 
- `hopnet`: HOPNet and the FIGNet/MeshGraphNet rollout reimplementations.

The instructions assume Linux x86-64, an NVIDIA GPU, and a driver compatible with the
listed CUDA runtimes. The wheel-based installation does not require a system `nvcc`.

## 1. Install Miniconda

Download the current Linux x86-64 Miniconda installer from the official Conda site,
run it, restart the shell, and confirm:

```bash
conda --version
nvidia-smi
```

## 2. Create the PIM environment

From `PIM_supplementary_code/`:

```bash
conda env create -f environments/pim_environment.yml
conda activate pim
python environments/verify_environment.py --environment pim
```

This installs Python 3.10, PyTorch 2.4.0 with CUDA 12.1, NumPy 2.0.1, MuJoCo 3.3.7,
SciPy, tqdm, trimesh, and matplotlib. PIM uses PyTorch scaled-dot-product attention and
requires the FlashAttention backend on CUDA. The verification script checks imports,
CUDA visibility, and a small attention operation; it does not run evaluation.

## 3. Create the rollout environment

```bash
conda env create -f environments/hopnet_environment.yml
conda activate hopnet

python -m pip install -r environments/hopnet_requirements.txt
python environments/verify_environment.py --environment hopnet
```

The requirements select PyTorch 2.4.1 CUDA 12.4 wheels and matching PyTorch Geometric
extension wheels. TopoNetX is pinned to the revision used by the working environment;
HOPNet imports TopoNetX directly and does not require TopoModelX.

## 4. Troubleshooting

- If `torch.cuda.is_available()` is false, first fix the NVIDIA driver or container GPU
  mapping. Installing a system CUDA toolkit does not repair an unavailable device.
- PyTorch Geometric extension wheels must match both PyTorch 2.4.1 and CUDA 12.4. An
  `undefined symbol` error usually means an extension was installed for another build.
- PIM's CUDA path intentionally fails instead of silently changing attention backends
  when FlashAttention is unavailable.
- HOPNet terminal rollout performs CPU-heavy collision processing. Limit concurrent
  workers if host RAM is constrained.

The environment names in all public instructions are `pim` and `hopnet`.

