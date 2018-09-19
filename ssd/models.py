from torch import nn
from torchvision.models import resnet34


class StdConv(nn.Module):
    """
    The convolution, batch normalization, and dropout layers gathered into
    single module.
    """
    def __init__(self, ni, nf, stride=2, kernel=3, padding=1, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv2d(ni, nf, kernel, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(nf)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class SSDConv(nn.Module):
    """
    A building block required to construct a Single-Shot Detector, two parallel
    convolutions predicting bounding boxes and classes of the objects.
    """
    def __init__(self, ni, n_classes, kernel=3, padding=1, k=1, bias=0):
        super().__init__()
        self.b_conv = nn.Conv2d(ni, 4 * k, kernel_size=kernel, padding=padding)
        self.c_conv = nn.Conv2d(
            ni, (n_classes + 1)*k, kernel_size=kernel, padding=padding)
        self.init(bias)

    def forward(self, x):
        return self.b_conv(x), self.c_conv(x)

    def init(self, bias):
        self.c_conv.bias.data.zero_().add_(bias)


class SSD(nn.Module):
    """
    Single-Shot Detector model.
    """
    def __init__(self, n_classes, dropout=0.25, bias=0, k=1,
                 backbone=resnet34, pretrained=True, flatten=True):

        super().__init__()
        model = backbone(pretrained=pretrained)
        children = list(model.children())

        self.k = k
        self.flatten = flatten
        self.backbone = nn.Sequential(*children[:-2])
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv1 = StdConv(512, 256, stride=1)
        self.conv2 = StdConv(256, 256)
        self.out = SSDConv(256, n_classes, k=k, bias=bias)

    def forward(self, x):
        x = self.backbone(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.out(x)
        if self.flatten:
            x = [flatten_conv(obj, self.k) for obj in x]
        return x


def flatten_conv(x, k):
    bs, nf, gx, gy = x.size()
    x = x.permute(0,2,3,1).contiguous()
    return x.view(bs, -1, nf//k)
