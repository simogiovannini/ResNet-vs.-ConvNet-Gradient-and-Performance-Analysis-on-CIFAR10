import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, skip_connection=False):
        super(ConvBlock, self).__init__()

        self.downsample = downsample
        self.skip_connection = skip_connection

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        if self.skip_connection:
            out += identity
        out = self.relu(out)
        return out


class ConvNet(nn.Module):
    def __init__(self, img_channels=3, num_classes=10, is_res_net=False):
        super(ConvNet, self).__init__()

        layers = [3, 4, 6, 3]
        self.in_channels = 64

        self.conv1 = nn.Conv2d(in_channels=img_channels, out_channels=self.in_channels, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2, is_res_net=is_res_net)
        self.layer3 = self._make_layer(256, layers[2], stride=2, is_res_net=is_res_net)
        self.layer4 = self._make_layer(512, layers[3], stride=2, is_res_net=is_res_net)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, blocks, stride=1, is_res_net=False):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        layers = [ConvBlock(self.in_channels, out_channels, stride, downsample, skip_connection=is_res_net)]
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(ConvBlock(self.in_channels, out_channels, skip_connection=is_res_net))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
