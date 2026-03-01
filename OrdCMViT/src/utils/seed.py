"""src/utils/seed.py — Reproducibility utilities"""
import os, random, numpy as np, torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Deterministic CUDA (may reduce performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Seed] Global seed set to {seed}")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[Device] VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[Device] Apple MPS")
    else:
        device = torch.device("cpu")
        print("[Device] CPU only — training will be slow")
    return device
