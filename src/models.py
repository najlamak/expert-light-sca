import torch, torch.nn as nn, torch.nn.functional as F

class ResidBlock1D(nn.Module):
    def __init__(self, c, k=9):
        super().__init__()
        self.conv1 = nn.Conv1d(c, c, k, padding=k//2)
        self.conv2 = nn.Conv1d(c, c, k, padding=k//2)
        self.bn1 = nn.GroupNorm(8, c); self.bn2 = nn.GroupNorm(8, c)
    def forward(self, x):
        y = F.gelu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.gelu(x + y)

class SCAEncoder(nn.Module):
    def __init__(self, in_ch=1, emb_dim=128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, 64, 11, stride=2, padding=5),
            nn.GroupNorm(8,64), nn.GELU(),
            nn.Conv1d(64,128,11, stride=2, padding=5),
            nn.GroupNorm(8,128), nn.GELU(),
        )
        self.blocks = nn.Sequential(ResidBlock1D(128), ResidBlock1D(128))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(128, emb_dim)
    def forward(self, x):
        x = self.stem(x); x = self.blocks(x)
        x = self.pool(x).flatten(1)
        z = self.head(x)
        return F.normalize(z, dim=-1)

class ProjectionMLP(nn.Module):
    def __init__(self, emb_dim=128, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.GELU(),
            nn.Linear(emb_dim, proj_dim)
        )
    def forward(self, z): return self.net(z)

class HWHead(nn.Module):
    def __init__(self, emb_dim=128, num_classes=9):
        super().__init__()
        self.fc = nn.Linear(emb_dim, num_classes)
    def forward(self, z): return self.fc(z)
