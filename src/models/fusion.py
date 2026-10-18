import torch
import torch.nn as nn

class DirectFusedMLP(nn.Module):
    def __init__(self, acoustic_input_dim = 768, linguistic_input_dim = 768, hidden_dim = 128, output_dim = 7, dropout = 0.0):
        super(DirectFusedMLP, self).__init__()

        combined_dim = acoustic_input_dim + linguistic_input_dim
        self.network = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p = dropout),

            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, acoustic_x, linguistic_x):
        acoustic_x = torch.flatten(acoustic_x, start_dim=1)
        linguistic_x = torch.flatten(linguistic_x, start_dim=1)

        fused_x = torch.concat((acoustic_x, linguistic_x), dim = 1)

        return self.network(fused_x)