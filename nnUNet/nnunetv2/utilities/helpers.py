import torch


def softmax_helper_dim0(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, 0)


def softmax_helper_dim1(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, 1)


def empty_cache(device: torch.device):
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        from torch import mps
        mps.empty_cache()
    else:
        pass


def is_oom_error(e: RuntimeError) -> bool:
    """
    True for both CUDA/MPS OOM ("... out of memory ...") and CPU allocation
    failures, which PyTorch reports as a RuntimeError from DefaultCPUAllocator
    (e.g. "can't allocate memory" / "not enough memory") rather than the
    "out of memory" wording used on GPU.
    """
    msg = str(e).lower()
    return 'out of memory' in msg or ('defaultcpuallocator' in msg and 'allocate memory' in msg)


class dummy_context(object):
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
