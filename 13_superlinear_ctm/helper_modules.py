import torch
import torch.nn as nn

class DropoutWithMask(nn.Module):
    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError("Dropout probability has to be between 0 and 1, but got {}".format(p))
        self.p = p
        self.inplace = inplace

    def forward(self, x):
        if not self.training or self.p == 0.0:
            mask = torch.ones_like(x, dtype=torch.int32, device=x.device)
            return x if self.inplace else x.clone(), mask

        # Generate dropout mask (1 where kept, 0 where dropped)
        mask = (torch.empty_like(x, device=x.device).uniform_() < self.p).to(dtype=torch.bool)

        if self.inplace:
            x.mul_(mask)
            x.mul_(1 / (1-self.p))
            return x, mask
        else:
            return x * mask * ((1/ (1-self.p))), mask
