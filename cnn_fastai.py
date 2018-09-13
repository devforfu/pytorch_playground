from fastai.conv_learner import *


PATH = '/home/ck/data/cifar10/'


stats = (np.array([ 0.4914 ,  0.48216,  0.44653]),
         np.array([ 0.24703,  0.24349,  0.26159]))


def get_data(sz, bs):
    tfms = tfms_from_stats(stats, sz, aug_tfms=[RandomFlip()], pad=sz//8)
    return ImageClassifierData.from_paths(PATH, val_name='valid', tfms=tfms, bs=bs)


class ConvLayer(nn.Module):

    def __init__(self, ni, nf, stride=2, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(
            ni, nf, kernel_size=kernel_size,
            stride=stride, bias=False, padding=1)
        self.bn = nn.BatchNorm2d(nf)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResNetLayer(ConvLayer):

    def forward(self, x):
        return x + super().forward(x)


class FastAIResNet(nn.Module):

    def __init__(self, layers, num_of_classes):
        super().__init__()
        self.conv = nn.Conv2d(3, 10, kernel_size=5, stride=1, padding=2)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                ConvLayer(x, y),
                ResNetLayer(y, y, stride=1),
                ResNetLayer(y, y, stride=1))
            for x, y in pairs(layers)])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(layers[-1], num_of_classes)

    def forward(self, x):
        x = self.conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def pairs(xs):
    current, *rest = xs
    for item in rest:
        yield current, item
        current = item


def main():
    bs = 256
    lr = 1e-2
    data = get_data(32, bs)
    net = FastAIResNet([10, 20, 40, 80, 160], 10)
    learn = ConvLearner.from_model_data(net, data)
    learn.fit(lr, 2, cycle_len=1, wds=1e-5)


if __name__ == '__main__':
    main()
