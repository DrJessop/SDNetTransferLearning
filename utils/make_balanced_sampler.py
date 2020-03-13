import torch


def make_sampler(train_set, p_labels):
    num_ones = 0
    for t in train_set:
        idx = int(t[0][0].item())
        num_ones += p_labels[idx].item()

    num_zeros = len(train_set) - num_ones
    counts = [num_zeros, num_ones]
    weights = [0] * len(train_set)

    for i, t in enumerate(train_set):
        idx = int(t[0][0].item())
        weights[i] = 1 / counts[p_labels[idx].item()]

    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(train_set))
    return sampler
