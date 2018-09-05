# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 04:05:02 2018

@author: Kezhong
"""
from torch import nn
import torch
import torch as t
from torch.nn import functional as F
from torch.utils.data import DataLoader
# import torchvision as tv
# import torchvision.transforms as transforms
# from torch.utils import data
# import os
# from PIL import Image
# from torchvision.datasets import ImageFolder
# #from loader import trainingmyImageFloder
# #from loader import testmyImageFloder
import loader
#from torch import optim
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import numpy as np
classes = ('Cat', 'Dog')

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# from torch.autograd import Variable
class ResidualBlock(nn.Module):
    '''
    实现子module: Residual Block
    '''

    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannel))
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)

class ResNet(nn.Module):
    '''
    实现主module：ResNet34
    ResNet34 包含多个layer，每个layer又包含多个residual block
    用子module来实现residual block，用_make_layer函数来实现layer
    '''

    def __init__(self, num_classes=1000):
        super(ResNet, self).__init__()
        # 前几层图像转换
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            #nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1))

        # 重复的layer，分别有3，4，6，3个residual block
        self.layer1 = self._make_layer(64, 128, 3)
        self.layer2 = self._make_layer(128, 256, 4, stride=2)
        self.layer3 = self._make_layer(256, 512, 6, stride=2)
        self.layer4 = self._make_layer(512, 512, 3, stride=2)

        # 分类用的全连接
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        '''
        构建layer,包含多个residual block
        '''
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
            nn.BatchNorm2d(outchannel))

        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)
        return self.fc(x)


#net = ResNet()
net = torchvision.models.resnet34(num_classes=2)
# #==========================================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer.zero_grad()


input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
output = criterion(input, target)
output.backward()
print('INPUT=' , input.size())
print('TARGET' , target)



# criterion = t.nn.MSELoss(reduce=False, size_average=False)
# learning_rate = 1e-4
# criterion = t.nn.MSELoss()
#===========================================================================
for epoch in range(2):  # loop over the dataset multiple times
    train_loader = loader.trainingmyImageFloder()
    #test_loader = loader.testmyImageFloder()
    #testloader = DataLoader(test_loader, batch_size=1, shuffle=False)
    trainloader = DataLoader(train_loader, batch_size=1, shuffle=False)
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader, 0):
        # get the inputs
        #inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        # print('HERE!!!!!!!!!!!!!!!!!!!!')
        # print('output=', outputs.size())
        labels = labels.view(1)  #
        labels = labels.type(torch.LongTensor)
        # print('target=', labels)
        # labels=labels.astype(np.int64)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

#========================================================
test_loader = loader.testmyImageFloder()
testloader = DataLoader(test_loader, batch_size=1, shuffle=False)
dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
# imshow(torchvision.utils.make_grid(images))
# print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

outputs = net(images)



correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        labels = labels.view(1)  #
        labels = labels.type(torch.LongTensor)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
