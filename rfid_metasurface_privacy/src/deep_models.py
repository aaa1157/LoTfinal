"""
深度学习攻击者模型

模型1: PhaseCNN - 普通 1D-CNN baseline
模型2: PhaseNet-Lite - Depthwise separable 1D convolution
模型3: TinyTCN - 轻量时序卷积网络 (dilation convolution)

输入: x shape = [batch, 1, 1800]
任务: motion / no_motion 二分类

依赖 torch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhaseCNN(nn.Module):
    """普通 1D-CNN baseline"""

    def __init__(self, seq_len: int = 1800, n_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)

        # 计算全连接层输入维度
        reduced_len = seq_len // 8  # 3次 pool
        self.fc1 = nn.Linear(128 * reduced_len, 64)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):
        # x: [B, 1, L]
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise separable 1D convolution"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size,
                                    padding=padding, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class PhaseNetLite(nn.Module):
    """轻量模型，使用 depthwise separable 1D convolution"""

    def __init__(self, seq_len: int = 1800, n_classes: int = 2):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv1d(1, 16, kernel_size=7, padding=3)
        self.conv2 = DepthwiseSeparableConv1d(16, 32, kernel_size=5, padding=2)
        self.conv3 = DepthwiseSeparableConv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)

        reduced_len = seq_len // 8
        self.fc1 = nn.Linear(64 * reduced_len, 32)
        self.fc2 = nn.Linear(32, n_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class TinyTCN(nn.Module):
    """轻量时序卷积网络，使用 dilation convolution"""

    def __init__(self, seq_len: int = 1800, n_classes: int = 2):
        super().__init__()
        channels = [1, 16, 32, 64, 64]
        dilations = [1, 2, 4, 8]
        kernel_size = 3

        layers = []
        for i in range(len(dilations)):
            padding = (kernel_size - 1) * dilations[i] // 2
            layers.append(nn.Conv1d(channels[i], channels[i+1], kernel_size,
                                     dilation=dilations[i], padding=padding))
            layers.append(nn.BatchNorm1d(channels[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))

        self.tcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels[-1], n_classes)

    def forward(self, x):
        x = self.tcn(x)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x


def count_parameters(model: nn.Module) -> int:
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ResNet1DLite(nn.Module):
    """轻量 1D ResNet，使用残差连接"""

    class ResBlock(nn.Module):
        def __init__(self, channels, kernel_size=3, dilation=1):
            super().__init__()
            padding = (kernel_size - 1) * dilation // 2
            self.conv1 = nn.Conv1d(channels, channels, kernel_size,
                                    padding=padding, dilation=dilation)
            self.bn1 = nn.BatchNorm1d(channels)
            self.conv2 = nn.Conv1d(channels, channels, kernel_size,
                                    padding=padding, dilation=1)
            self.bn2 = nn.BatchNorm1d(channels)

        def forward(self, x):
            residual = x
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += residual
            return F.relu(out)

    def __init__(self, seq_len: int = 1800, n_classes: int = 2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.layer1 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            self.ResBlock(64),
            nn.MaxPool1d(2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            self.ResBlock(128),
            nn.AdaptiveAvgPool1d(1),
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class DualBranchNet(nn.Module):
    """
    双分支网络：时域 + 频域融合

    输入: x shape = [batch, 1, 1800]
    分支1: 时域 phase sequence -> 1D CNN
    分支2: 频域 FFT magnitude -> 1D CNN
    融合后做 motion/no_motion 二分类
    """

    def __init__(self, seq_len: int = 1800, n_classes: int = 2):
        super().__init__()
        self.seq_len = seq_len
        fft_len = seq_len // 2 + 1

        # 时域分支
        self.time_branch = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # 频域分支
        self.freq_branch = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(32 + 32, 32)
        self.fc2 = nn.Linear(32, n_classes)

    def forward(self, x):
        # x: [B, 1, L]
        # 时域分支
        time_feat = self.time_branch(x).squeeze(-1)  # [B, 32]

        # 频域分支
        x_fft = torch.fft.rfft(x, dim=-1)
        x_mag = torch.abs(x_fft)  # [B, 1, L//2+1]
        freq_feat = self.freq_branch(x_mag).squeeze(-1)  # [B, 32]

        # 融合
        combined = torch.cat([time_feat, freq_feat], dim=-1)  # [B, 64]
        out = self.dropout(F.relu(self.fc1(combined)))
        out = self.fc2(out)
        return out


def get_model(name: str, seq_len: int = 1800, n_classes: int = 2) -> nn.Module:
    """获取模型实例"""
    models = {
        'PhaseCNN': PhaseCNN,
        'PhaseNetLite': PhaseNetLite,
        'TinyTCN': TinyTCN,
        'ResNet1DLite': ResNet1DLite,
        'DualBranchNet': DualBranchNet,
    }
    if name not in models:
        raise ValueError(f"Unknown model: {name}. Available: {list(models.keys())}")
    return models[name](seq_len=seq_len, n_classes=n_classes)
