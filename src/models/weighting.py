from torch.utils.data import WeightedRandomSampler

def class_weighting(class_distb):
    no_classes = len(class_distb)
    total_samples = class_distb.sum()

    class_weights = total_samples / (no_classes * class_distb)
    return class_weights

def weight_sampling(class_distb, class_labels):
    class_weights = 1.0 / class_distb
    sample_weights = class_weights[class_labels]

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    return sampler