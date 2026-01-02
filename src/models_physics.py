import torch
import torch.nn as nn

class PhysicsAwareStem(nn.Module):
    """
    Multi-branch stem that processes physically meaningful band groups separately,
    then fuses features into a single tensor.

    Input:  x [B, C=13, H, W]
    Output: f [B, F, H/2, W/2]  (by default downsample like ResNet's conv1+maxpool-ish behavior)
    """
    def __init__(self, band_groups, out_per_group=16):
        super().__init__()
        self.band_groups = band_groups

        def make_branch(in_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_per_group, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_per_group),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_per_group, out_per_group, kernel_size=3, stride=2, padding=1, bias=False),  # downsample
                nn.BatchNorm2d(out_per_group),
                nn.ReLU(inplace=True),
            )

        self.branches = nn.ModuleDict({
            name: make_branch(len(idxs)) for name, idxs in band_groups.items()
        })

        # Fuse = concat + 1x1 conv to a standard width
        fused_in = out_per_group * len(band_groups)
        self.fuse = nn.Sequential(
            nn.Conv2d(fused_in, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        feats = []
        for name, idxs in self.band_groups.items():
            xi = x[:, idxs, :, :]
            feats.append(self.branches[name](xi))
        f = torch.cat(feats, dim=1)
        return self.fuse(f)


class BasicBlock(nn.Module):
    """A minimal ResNet BasicBlock (same spirit as torchvision)."""
    expansion = 1
    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = self.relu(out + identity)
        return out


class PhysicsAwareResNet18(nn.Module):
    """
    Physics-aware version of a ResNet18-style network:
    - PhysicsAwareStem handles band grouping + early processing
    - Standard ResNet blocks for deeper feature learning
    """
    def __init__(self, band_groups, num_classes=10):
        super().__init__()
        self.stem = PhysicsAwareStem(band_groups, out_per_group=16)

        # After stem, we have 64 channels at H/2, W/2
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, inplanes, planes, blocks, stride):
        layers = [BasicBlock(inplanes, planes, stride=stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(planes, planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)              # [B, 64, H/2, W/2]
        x = self.layer1(x)            # [B, 64, ...]
        x = self.layer2(x)            # [B, 128, ...]
        x = self.layer3(x)            # [B, 256, ...]
        x = self.layer4(x)            # [B, 512, ...]
        x = self.pool(x).flatten(1)   # [B, 512]
        return self.fc(x)