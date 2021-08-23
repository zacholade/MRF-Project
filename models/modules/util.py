import torch


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