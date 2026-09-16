import torch 
from torch import nn

class FeedForward(nn.Module):
    def __init__(self, d_model, dff):
        super().__init__()

        self.d_model = d_model
        self.dff = dff
        self.linear_layer1 = nn.Linear(d_model, dff)
        self.linear_layer2 = nn.Linear(dff, d_model)

    def forward(self, x):

        return self.linear_layer2(torch.relu(self.linear_layer1(x)))