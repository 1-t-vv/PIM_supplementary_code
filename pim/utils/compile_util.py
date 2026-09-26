from __future__ import annotations

import warnings
from typing import TypeVar

import torch


ModuleT = TypeVar("ModuleT", bound=torch.nn.Module)


def compile_training_model(
    model: ModuleT,
    *,
    enabled: bool,
    backend: str = "inductor",
    mode: str = "default",
    fullgraph: bool = False,
    dynamic: bool = False,
    suppress_errors: bool = True,
) -> torch.nn.Module:
    """Compile the training wrapper while preserving the caller's eager model.

    The caller should retain its original Predictor reference for validation and
    checkpoints. ``torch.compile`` returns an OptimizedModule whose state-dict
    keys contain ``_orig_mod.``; saving the original Predictor avoids changing
    the checkpoint format.

    For distributed training this function must receive an already-created DDP
    wrapper. PyTorch can then use DDPOptimizer to split compiled graphs along
    gradient bucket boundaries and retain backward/all-reduce overlap.
    """
    if not enabled:
        return model
    if not hasattr(torch, "compile"):
        warnings.warn("torch.compile is unavailable; using eager training.")
        return model
    if not backend:
        raise ValueError("torch_compile_backend must not be empty")
    if not mode:
        raise ValueError("torch_compile_mode must not be empty")

    # Compilation is lazy: many backend failures occur on the first forward,
    # not in torch.compile() itself. suppress_errors lets Dynamo fall back to
    # eager execution for an unsupported graph instead of losing a long DDP
    # run. It does not suppress model/runtime correctness errors.
    torch._dynamo.config.suppress_errors = bool(suppress_errors)
    try:
        return torch.compile(
            model,
            backend=str(backend),
            mode=str(mode),
            fullgraph=bool(fullgraph),
            dynamic=bool(dynamic),
        )
    except Exception as exc:
        if not suppress_errors:
            raise
        warnings.warn(
            f"torch.compile setup failed; using eager training. Cause: {exc}"
        )
        return model
