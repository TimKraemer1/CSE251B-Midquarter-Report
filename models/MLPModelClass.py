import torch.nn as nn

class MLPModel(nn.Module):
    def __init__(self, input_features, output_features):
        super(MLPModel, self).__init__()

        self.mlp = nn.Sequential(
            # Input Layer ( 120 -> 1024 )
            nn.Linear(input_features, 1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),

            # Hidden Layer ( 1024 -> 512  )
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),

            # Hidden Layer ( 512 -> 256 )
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),

            # Hidden Layer ( 256 -> 128 )
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),

            # Output Layer ( 128 -> 60*2 )
            nn.Linear(128, output_features)
        )
    
    def forward(self, data):
        x = data.x  # (batch_size * 50, 50, 6)
        batch_size = x.shape[0] // 50

        x = x.view(batch_size, 50, 50, 6)  # (batch, agents, timesteps, features)
        ego_x = x[:, 0, :, :]  # (batch, 50, 6)
        ego_x = ego_x.reshape(batch_size, -1)  # (batch, 300)

        x = self.mlp(ego_x)  # (batch, 120)
        return x.view(batch_size, 60, 2)