import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=32, kernel_size=(1, 3), padding=(0, 1)),  # (B, 32, 1, 50)
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, kernel_size=(1, 3), padding=(0, 1)),  # (B, 64, 1, 50)
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.AdaptiveAvgPool2d((1, 10))  # (B, 64, 1, 10)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),               # (B, 64*1*10)
            nn.Linear(64 * 10, 256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 120)
        )

    def forward(self, data):
        x = data.x  # (batch_size * 50, 50, 6)
        batch_size = x.shape[0] // 50

        x = x.view(batch_size, 50, 50, 6)  # (B, 50 agents, 50 timesteps, 6 features)
        ego = x[:, 0, :, :]               # (B, 50, 6)

        ego = ego.permute(0, 2, 1).unsqueeze(2)  # (B, 6, 1, 50)

        x = self.cnn(ego)     # (B, 64, 1, 10)
        x = self.fc(x)        # (B, 120)
        return x.view(batch_size, 60, 2)