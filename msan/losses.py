import torch.nn as nn


class MSANLoss(nn.Module):
    """
    Consistent with original MSAN core loss:
      total_loss = l1_loss + mse_loss
    """

    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()

    def forward(self, output, target):
        mse_loss = self.mse(output, target)
        l1_loss = self.l1(output, target)
        total_loss = l1_loss + mse_loss
        return total_loss, mse_loss, l1_loss

