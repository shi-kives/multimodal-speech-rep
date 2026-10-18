import torch
from torch.utils.data import Dataset

class SpeechEmbeddingsDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.tensor(embeddings, dtype = torch.float32)
        self.labels = torch.tensor(labels, dtype = torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


class FusedDataset(Dataset):
    def __init__(self, acoustic_embeddings, linguistic_embeddings, labels):
        self.acoustic_embeddings = torch.tensor(acoustic_embeddings, dtype = torch.float32)
        self.linguistic_embeddings = torch.tensor(linguistic_embeddings, dtype = torch.float32)
        self.labels = torch.tensor(labels, dtype = torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.acoustic_embeddings[idx], self.linguistic_embeddings[idx], self.labels[idx]
