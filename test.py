import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset_npz import NPZFusionDataset
from msan import MSAN


def parse_args():
    p = argparse.ArgumentParser("MSAN standalone tester")
    p.add_argument("--test_dir", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--spectral_num", type=int, default=8)
    p.add_argument("--spatial_num", type=int, default=1)
    p.add_argument("--save_dir", type=str, default="predictions")
    return p.parse_args()


def run():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = NPZFusionDataset(args.test_dir)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    model = MSAN(spectral_num=args.spectral_num, spatial_num=args.spatial_num).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            lms = batch["lms"].to(device)
            pan = batch["pan"].to(device)
            out = model(lms, pan).cpu().numpy()
            np.save(save_dir / f"pred_{i:05d}.npy", out)
    print(f"Saved predictions to: {save_dir}")


if __name__ == "__main__":
    run()

