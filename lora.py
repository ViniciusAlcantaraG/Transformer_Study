import torch
from torch import nn

class LoRA_Layer(nn.Module):
    def __init__(self, input_dim, output_dim, rank, alpha):
        super().__init__()

        std_dev = 1/torch.sqrt(torch.tensor(rank).float())
        self.A = nn.Parameter(torch.randn(input_dim, rank) * std_dev)
        self.B = nn.Parameter(torch.zeros(rank, output_dim))
        self.alpha = alpha

    def forward(self, x):

        x = self.alpha * (x @ self.A @ self.B)

class LinearWithLoRA(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()

        self.linear = linear
        self.lora = LoRA_Layer(linear.in_features, linear.out_features, rank, alpha)

        def forward(self, x):

            return self.linear(x) + self.lora(x)