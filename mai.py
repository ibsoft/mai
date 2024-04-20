import torch
import torch.nn as nn


class maiNeuralNetWork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_prob):
        super(maiNeuralNetWork, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.dropout1(self.l1(x)))
        out = self.relu(self.dropout2(self.l2(out)))
        out = self.l3(out)
        # No activation and no softmax at the end
        return out
