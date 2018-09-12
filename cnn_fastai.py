from fastai.conv_learner import *


PATH = '/home/ck/data/cifar10/'


stats = (np.array([ 0.4914 ,  0.48216,  0.44653]),
         np.array([ 0.24703,  0.24349,  0.26159]))


def get_data(sz, bs):
    tfms = tfms_from_stats(stats, sz, aug_tfms=[RandomFlip()], pad=sz//8)
    return ImageClassifierData.from_paths(PATH, val_name='valid', tfms=tfms, bs=bs)


class SimpleNet(nn.Module):

    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(layers[i], layers[i + 1])
            for i in range(len(layers) - 1)
        ])

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for l in self.layers:
            x = F.relu(l(x))
        return F.log_softmax(x, dim=-1)


def main():
    bs = 256
    lr = 1e-2
    data = get_data(32, bs)
    net = SimpleNet([32*32*3, 40, 10])
    learn = ConvLearner.from_model_data(net, data)
    learn.fit(lr, 2, cycle_len=1)


if __name__ == '__main__':
    main()
