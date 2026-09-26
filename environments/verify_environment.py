from __future__ import annotations

import argparse
import importlib
import sys


def require(module_name: str) -> object:
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        raise RuntimeError(f"Cannot import required module {module_name!r}") from exc


def verify_pim() -> None:
    torch = require("torch")
    require("numpy")
    require("scipy")
    require("tqdm")
    require("mujoco")
    require("trimesh")
    if not torch.cuda.is_available():
        raise RuntimeError("The pim environment requires a visible NVIDIA CUDA device")
    device = torch.device("cuda")
    query = torch.zeros((1, 1, 1, 64), device=device, dtype=torch.float16)
    key = torch.zeros((1, 1, 32, 64), device=device, dtype=torch.float16)
    with torch.nn.attention.sdpa_kernel(
        backends=[torch.nn.attention.SDPBackend.FLASH_ATTENTION]
    ):
        torch.nn.functional.scaled_dot_product_attention(query, key, key)
    torch.cuda.synchronize()
    print(f"pim environment OK: Python {sys.version.split()[0]}, torch {torch.__version__}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")


def verify_hopnet() -> None:
    torch = require("torch")
    require("numpy")
    require("scipy")
    require("trimesh")
    require("torch_geometric")
    require("torch_scatter")
    require("torch_sparse")
    require("toponetx")
    if not torch.cuda.is_available():
        raise RuntimeError("The hopnet environment requires a visible NVIDIA CUDA device")
    print(f"hopnet environment OK: Python {sys.version.split()[0]}, torch {torch.__version__}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", choices=("pim", "hopnet"), required=True)
    args = parser.parse_args()
    verify_pim() if args.environment == "pim" else verify_hopnet()


if __name__ == "__main__":
    main()

