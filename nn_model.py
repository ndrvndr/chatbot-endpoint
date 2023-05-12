import torch.nn as nn


class neural_network(nn.Module):
    def __init__(self, input_layer, hidden_layer, output_layer):
        super(neural_network, self).__init__()

        self.l1 = nn.Linear(input_layer, hidden_layer)
        self.l2 = nn.Linear(hidden_layer, hidden_layer)
        self.l3 = nn.Linear(hidden_layer, output_layer)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)

        return out
