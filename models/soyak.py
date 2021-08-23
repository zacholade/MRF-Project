import torch
from torch import nn

from models.modules.cbam import CBAM


class Soyak(nn.Module):
    """
    https://ieeexplore.ieee.org/document/8759502
    """
    def __init__(self, patch_size: int, seq_len: int):
        super().__init__()
        self.patch_size = patch_size
        self.cbam = CBAM(seq_len, reduction_ratio=1, no_spatial=True)
        self.conv1 = nn.Conv2d(32, 32, 3, padding='same')
        self.conv2 = nn.Conv2d(32, 64, 3, padding='same')
        self.conv3 = nn.Conv2d(64, 128, 3, padding='same')
        self.conv4 = nn.Conv2d(128, 64, 3, padding='same')
        self.dense = nn.Linear(64, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.shape[0]

        # Pass through attention module. Calculate scales to pick top 32 channels.
        _, scale = self.cbam(x, return_scale=True)
        scale = scale.contiguous().view(batch_size * self.patch_size * self.patch_size, -1)

        # Select the top 32 channels.
        top_n_indices = torch.topk(scale, 32, dim=1).indices
        x = x.contiguous().view(batch_size * self.patch_size * self.patch_size, -1)
        x = batched_index_select(x, 1, top_n_indices).view(batch_size, 32, self.patch_size, self.patch_size)

        # Pass through conv layers.
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))

        # Classification. Output dim is 2 for t1 t2.
        x = self.dense(x.transpose(1, 3))
        return x


def batched_index_select(input, dim, index):
    """
    Similar to torch.index_select but works on batches.
    https://discuss.pytorch.org/t/batched-index-select/9115/5
    """
    views = [input.shape[0]] + [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(input, dim, index)
