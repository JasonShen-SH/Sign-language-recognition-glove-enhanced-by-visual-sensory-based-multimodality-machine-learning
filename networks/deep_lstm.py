import torch
import torch.nn as nn

class DeepLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(DeepLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm_layers = nn.ModuleList()
        self.lstm_layers.append(nn.LSTM(input_size, hidden_size, batch_first=True))

        # add more LSTM layers
        for _ in range(1, num_layers):
            self.lstm_layers.append(nn.LSTM(hidden_size, hidden_size, batch_first=True))

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # initialize
        # batch_size = x.size(0)
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)

        for lstm in self.lstm_layers:
            out, (hn, cn) = lstm(x, (h0, c0))
            x = out  

        out = out[:, -1, :]

        out = self.fc(out)

        return out