import torch
import torch.nn as nn

class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim = 768, hidden_dim = 128, output_dim = 7):
        super(MultiLayerPerceptron, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        return self.network(x)