from torch import nn

from models.modules.cbam import CBAM


class Soyak(nn.Module):
    """
    https://ieeexplore.ieee.org/document/8759502
    """
    def __init__(self, patch_size: int, seq_len: int):
        super().__init__()
        self.patch_size = patch_size
        self.cbam = CBAM(seq_len, reduction_ratio=1)
        self.relu = nn.ReLU()



    def forward(self, x):
        print(x.shape)
        x = self.cbam(x)
        return x