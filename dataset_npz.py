from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset


class NPZFusionDataset(Dataset):
    """
    Standalone dataset for MSAN sharing.
    Expected each .npz file contains keys: lms, pan, gt.
    Shapes:
      - lms: [C, H, W]
      - pan: [1, H, W]
      - gt:  [C, H, W]
    """

    def __init__(self, root: str):
        self.root = Path(root)
        self.files = sorted(self.root.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No npz files found in: {self.root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        data = np.load(self.files[idx])
        lms = torch.from_numpy(data["lms"]).float()
        pan = torch.from_numpy(data["pan"]).float()
        gt = torch.from_numpy(data["gt"]).float()
        return {"lms": lms, "pan": pan, "gt": gt}

