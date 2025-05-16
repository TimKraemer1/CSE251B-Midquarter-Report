# Pipeline used from midquarter checkpoint for simplicity and consistency
# -------------------------------------------------------------------

# Pytorch Specific Libraries
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

# Data
import numpy as np

class TrajectoryDatasetTrain(Dataset):
    def __init__(self, data, scale=10.0, augment=True):  
        self.data = data # Shape (N, 50, 110, 6)
        self.scale = scale # Normalization scale
        self.augment = augment # Data augmentation flag

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        scene = self.data[idx]
        # Getting 50 historical timestamps and 60 future timestamps
        hist = scene[:, :50, :].copy()    # (agents=50, time_seq=50, 6)
        future = torch.tensor(scene[0, 50:, :2].copy(), dtype=torch.float32)  # (60, 2)

        # ---------- Agent Velocity Pre-computation ----------
        vx = hist[:, 49, 2]
        vy = hist[:, 49, 3]
        speed = np.sqrt(vx ** 2 + vy ** 2)

        # Top 4 fastest agents excluding ego car
        ego_agent = hist[0:1] #Shape (1, 50, 6)

        non_ego_speeds = speed[1:]
        top4_idx = np.argsort(non_ego_speeds)[-4:] + 1
        top4_idx = hist[top4_idx] # Shape (4, 50, 6)
        # hist = np.concatenate([ego_agent, top4_idx], axis=0)  # Shape (5, 50, 6)
        
        # ---------- End of Agent Velocity Pre-computation ----------
        
        # Data augmentation(only for training)
        if self.augment:
            if np.random.rand() < 0.5:
                theta = np.random.uniform(-np.pi, np.pi)
                R = np.array([[np.cos(theta), -np.sin(theta)],
                              [np.sin(theta),  np.cos(theta)]], dtype=np.float32)
                # Rotate the historical trajectory and future trajectory
                hist[..., :2] = hist[..., :2] @ R
                hist[..., 2:4] = hist[..., 2:4] @ R
                future = future @ R
            if np.random.rand() < 0.5:
                hist[..., 0] *= -1
                hist[..., 2] *= -1
                future[:, 0] *= -1

        # Use the last timeframe of the historical trajectory as the origin
        origin = hist[0, 49, :2].copy()  # (2,)
        hist[..., :2] = hist[..., :2] - origin
        future = future - origin

        # Normalize the historical trajectory and future trajectory
        hist[..., :4] = hist[..., :4] / self.scale
        future = future / self.scale

        data_item = Data(
            x=torch.tensor(hist, dtype=torch.float32),
            y=future.type(torch.float32),
            origin=torch.tensor(origin, dtype=torch.float32).unsqueeze(0),
            scale=torch.tensor(self.scale, dtype=torch.float32),
        )

        return data_item
    

class TrajectoryDatasetTest(Dataset):
    def __init__(self, data, scale=10.0):
        """
        data: Shape (N, 50, 110, 6) Testing data
        scale: Scale for normalization (suggested to use 10.0 for Argoverse 2 data)
        """
        self.data = data
        self.scale = scale

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Testing data only contains historical trajectory
        scene = self.data[idx]  # (50, 50, 6)
        hist = scene.copy()
        
        origin = hist[0, 49, :2].copy()
        hist[..., :2] = hist[..., :2] - origin
        hist[..., :4] = hist[..., :4] / self.scale

        data_item = Data(
            x=torch.tensor(hist, dtype=torch.float32),
            origin=torch.tensor(origin, dtype=torch.float32).unsqueeze(0),
            scale=torch.tensor(self.scale, dtype=torch.float32),
        )
        return data_item