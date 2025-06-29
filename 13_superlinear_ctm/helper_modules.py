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
            mask = torch.ones_like(x, dtype=x.dtype, device=x.device)
            return x if self.inplace else x.clone(), mask

        # Generate dropout mask (1 where kept, 0 where dropped)
        mask = torch.empty_like(x, dtype=x.dtype, device=x.device).bernoulli_(1 - self.p)

        # Scale mask in-place
        mask.div_(1 - self.p)

        if self.inplace:
            x.mul_(mask)
            return x, mask
        else:
            return x * mask, mask
