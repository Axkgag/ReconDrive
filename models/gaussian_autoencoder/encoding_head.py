import torch.nn as nn


class EncodingHead(nn.Module):
    """Lightweight MLP: 86-dim Gaussian feature → 32-dim voxel feature."""

    def __init__(self, in_dim=86, hidden_dims=(128, 64), out_dim=32):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """x: [M, 86] → [M, 32]"""
        return self.net(x)


class DecodingHead(nn.Module):
    """Symmetric MLP: 32-dim voxel feature → K × 86-dim Gaussian parameters."""

    def __init__(self, in_dim=32, hidden_dims=(128, 256), K=4, gauss_dim=86):
        super().__init__()
        self.K = K
        self.gauss_dim = gauss_dim
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True)]
            prev = h
        layers.append(nn.Linear(prev, K * gauss_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """x: [M, 32] → [M, K, 86]"""
        M = x.shape[0]
        return self.net(x).reshape(M, self.K, self.gauss_dim)
