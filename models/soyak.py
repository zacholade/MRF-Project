import torch
from torch import nn

from models.modules.cbam import CBAM, CBAMChannelReduction
from models.modules.util import batched_index_select


class Soyak(nn.Module):
    """
    https://ieeexplore.ieee.org/document/8759502
    """
    def __init__(self, patch_size: int, seq_len: int):
        super().__init__()
        self.patch_size = patch_size
        self.cbam_channel_reduction = CBAMChannelReduction(seq_len, reduction=32)
        self.cbam = CBAM(seq_len, reduction_ratio=1, no_spatial=True)
        self.conv1 = nn.Conv2d(32, 32, 3, padding='same')
        self.conv2 = nn.Conv2d(32, 64, 3, padding='same')
        self.conv3 = nn.Conv2d(64, 128, 3, padding='same')
        self.conv4 = nn.Conv2d(128, 64, 3, padding='same')
        self.dense = nn.Linear(64, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.shape[0]

        x, scale = self.cbam_channel_reduction(x)

        # Pass through conv layers.
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))

        # Classification. Output dim is 2 for t1 t2.
        x = self.dense(x.transpose(1, 3))
        return x, scale


