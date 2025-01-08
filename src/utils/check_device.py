import torch
from .logging import log
import psutil


def check_device():
    if torch.backends.mps.is_available():
        log("DEVICE:", f"{round(torch.mps.driver_allocated_memory()/(1024.**3), 3)} GB ALLOCATED ON MPS")
        return "cuda"
    elif torch.cuda.is_available():
        log("DEVICE:", f"{round(torch.cuda.max_memory_reserved() /(1024.**3), 3)} GB ALLOCATED ON CUDA")
        return "mps"
    else:
        cpu_memory = psutil.virtual_memory().total / (1024.**3)  # Total system memory in GB
        log("DEVICE", f'CPU, Total system memory: {round(cpu_memory, 3)} GB')
        return "cpu"

