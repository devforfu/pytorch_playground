from os.path import expanduser

import torch
import torchvision
import torchvision.transforms as transforms


PATH = '~/data/cifar10'

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(
    root=expanduser(PATH),
    train=True,
    download=True,
    transform=transform)

trainloader = torch.utils.data.DataLoader(
    trainset, 
    batch_size=4,
    shuffle=True,
    num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root=expanduser(PATH),
    train=False,
    download=True,
    transform=transform)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=4,
    shuffle=False,
    num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse',
           'ship', 'truck')


