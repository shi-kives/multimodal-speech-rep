import torch
from torch.utils.data import SubsetRandomSampler

def generate_sampler(dataset,  percentage):
    dataset_length = len(dataset)
    total_samples =  int(dataset_length * percentage)
    print(f"sampling {total_samples} / {dataset_length} samples in train.")

    indices = torch.randperm(dataset_length)[:total_samples].tolist()

    sampler = SubsetRandomSampler(indices)
    
    return sampler