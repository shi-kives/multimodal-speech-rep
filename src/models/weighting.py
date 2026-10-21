from torch.utils.data import WeightedRandomSampler
import torch

class_distb = torch.tensor([471, 140, 106, 670, 1816, 261, 497])
def class_weighting():
    no_classes = len(class_distb)
    total_samples = class_distb.sum()

    class_weights = total_samples / (no_classes * class_distb)
    return class_weights

def weight_sampling(class_labels):
    class_weights = 1.0 / class_distb
    sample_weights = class_weights[class_labels]

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    return sampler