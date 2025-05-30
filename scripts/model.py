# scripts/model.py

import torch
import torch.nn as nn

class CNNLSTMModel(nn.Module):
    def __init__(self, input_dim, cnn_filters=64, lstm_hidden=512, output_days=31):
        super(CNNLSTMModel, self).__init__()
        
        self.cnn = nn.Conv1d(
            in_channels=input_dim,  # feature 수
            out_channels=cnn_filters,
            kernel_size=2,
            stride=1
        )
        
        self.lstm = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=lstm_hidden,
            batch_first=True
        )
        
        self.fc = nn.Linear(lstm_hidden, output_days)

    def forward(self, x):
        # x: [B, T, F] → CNN 입력 위해 [B, F, T]로 변경
        x = x.permute(0, 2, 1)

        x = self.cnn(x)               # [B, C, T']
        x = x.permute(0, 2, 1)        # LSTM 입력 위해 [B, T', C]

        _, (h_n, _) = self.lstm(x)    # h_n: [1, B, H]
        out = self.fc(h_n.squeeze(0)) # [B, output_days]

        return out
