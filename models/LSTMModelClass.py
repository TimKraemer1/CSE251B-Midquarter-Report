# Pytorch Specific Libraries
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=256, output_dim=60 * 2):
        super(LSTMModel, self).__init__()

        # LSTM, input_dim=6 (features), hidden_dim=256
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        # LSTM dropout with 0.3 probability
        self.dropout = nn.Dropout(p=0.3)

        # Hidden Layer, 256 -> 128 -> 256 -> 60*2
        self.fc1 = nn.Linear(hidden_dim, 128) 
        self.fc2 = nn.Linear(128, 256)
        self.relu = nn.ReLU()
        self.output = nn.Linear(256, output_dim)

    def forward(self, data):
        x = data.x
        x= x.reshape(-1, 50, 50, 6)  # (batch_size, num_agents, seq_len, input_dim)
        x = x[:, 0, :, :]  # (batch_size, seq_len, input_dim) (ego car only)

        lstm_out, _ = self.lstm(x)

        last_hidden = lstm_out[:, -1, :]  # final LSTM output for each sequence
        
        x = self.dropout(self.relu(self.fc1(last_hidden)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.output(x)
        return x.view(-1, 60, 2)