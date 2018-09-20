from torch import nn
from torch.nn import functional as F


class Flatten(nn.Module):
    """
    Pseudo-layer converting convolution output into flat format compatible with
    linear layers.
    """
    def forward(self, x):
        return x.view(x.size(0), -1)


class LinearConv3x3(nn.Module):
    """
    A convolution with 3x3 kernel and with linear activations.

    Params:
        ni: Number of input channels.
        nf: Number of output channels.
        stride: Convolution stride.
        padding: Convolution padding.

    """
    def __init__(self, ni, nf, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(ni, nf, 3, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(nf)

    def forward(self, x):
        return self.bn(self.conv(x))


class IdentityBlock(nn.Module):
    """
    Basic building block for small ResNet models.

    The block consists of two convolutions with shortcut connections between
    between input and output. Note that this type of block is a simple version
    usually used for "shallow" ResNets. Deeper networks use bottleneck design
    for performance considerations.

    Params:
        ni: Number of input channels.
        nf: Number of output channels. If None, then `ni` value is used.
            Otherwise, the block includes downsampling convolution to convert
            input tensor into shape compatible with output before applying
            addition.
        stride: The stride value of the first and downsampling convolutions.

    """
    def __init__(self, ni, nf=None, stride=1):
        super().__init__()

        nf = ni if nf is None else nf

        self.conv1 = LinearConv3x3(ni, nf, stride=stride)
        self.conv2 = LinearConv3x3(nf, nf)
        if ni != nf:
            self.downsample = nn.Sequential(
                nn.Conv2d(ni, nf, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(nf))

    def forward(self, x):
        out = self.conv1(x)
        out = F.leaky_relu(out)
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.leaky_relu(out)
        return out

    def shortcut(self, x):
        if hasattr(self, 'downsample'):
            return self.downsample(x)
        return x


class ResNet(nn.Module):
    """
    Custom ResNet classification architecture with identity blocks.
    """
    def __init__(self, num_of_classes):
        super().__init__()
        self.conv = LinearConv3x3(1, 10, padding=2)
        self.blocks = nn.ModuleList([
            IdentityBlock(10, 20, stride=2),
            IdentityBlock(20, 40, stride=2),
            IdentityBlock(40, 80, stride=2)
        ])
        self.pool = nn.AvgPool2d(4)
        self.flatten = Flatten()
        self.fc = nn.Linear(80, num_of_classes)
        init(self)

    def forward(self, x):
        x = F.leaky_relu(self.conv(x))
        for block in self.blocks:
            x = block(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def init(m):
    if hasattr(m, 'children'):
        for child in m.children():
            init(child)
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
