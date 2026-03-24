import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset_npz import NPZFusionDataset
from msan import MSAN, MSANLoss


def parse_args():
    p = argparse.ArgumentParser("MSAN standalone trainer")
    p.add_argument("--train_dir", type=str, required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--spectral_num", type=int, default=8)
    p.add_argument("--spatial_num", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--save_dir", type=str, default="checkpoints")
    p.add_argument("--no_amp", action="store_true", default=False)
    return p.parse_args()


def run():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available() and (not args.no_amp)

    train_ds = NPZFusionDataset(args.train_dir)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
    )

    model = MSAN(spectral_num=args.spectral_num, spatial_num=args.spatial_num).to(device)
    loss_fn = MSANLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            lms = batch["lms"].to(device, non_blocking=True)
            pan = batch["pan"].to(device, non_blocking=True)
            gt = batch["gt"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(lms, pan)
                total_loss, mse_loss, l1_loss = loss_fn(out, gt)

            if not torch.isfinite(total_loss):
                continue

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running += total_loss.item()

        scheduler.step()
        avg_loss = running / max(1, len(train_loader))
        print(f"[Epoch {epoch:03d}] total={avg_loss:.6f} (last mse={mse_loss.item():.6f}, l1={l1_loss.item():.6f})")

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "args": vars(args),
        }
        torch.save(ckpt, save_dir / f"msan_epoch_{epoch:03d}.pth")


if __name__ == "__main__":
    run()

