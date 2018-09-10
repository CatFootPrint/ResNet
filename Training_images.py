# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 16:25:02 2018
@author: Kezhong Zhang
"""
############################################################
# Load the packages
from torch import nn
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
# import matplotlib.pyplot as plt
# import numpy as np
import time
##############################################################
# Definition
# ============================================================
# Definition of the model type
# Optional models are
# ResNet152
# AlexNet
# VGG11
# VGG13
# VGG16
# VGG19
# ResNet18
# ResNet34
# ResNet50
# ResNet101
# ResNet152
# SqueezeNet1_0
# SqueezeNet1_1
# Densenet121
# Densenet169
# Densenet201
# Densenet161
# Inception_v3
model_name = 'ResNet34'
# ============================================================
# Definition of data type
# data_type = 'Dog_vs_Cat'
data_type = 'Cyclic'
# data_type = 'CIFAR10'
# ============================================================
# ============================================================
# 1. Definition of variables
# Number of iterations in training stage
num_iterations = 4
# Size of patches
define_batch_size = 1
# learning rate in training stage
learning_rate = 0.00001
# ============================================================
# 2. Definition of paths. The optional paths are listed
# ------------------------------------------------------------
# path of the training and test data
if data_type == 'Dog_vs_Cat':
    training_file_path = './dogcat'
    test_file_path = './dogcat_test'
    # Definition of classes
    classes = ('Cat', 'Dog')
elif data_type == 'Cyclic':
    training_file_path = './data_cyclic/training_cyclic'
    test_file_path = './data_cyclic/test_cyclic'
    # Definition of classes
    classes = ('BPSK', 'QPSK', '8QAM', '64QAM')
elif data_type == 'CIFAR10':
    training_file_path = './CIFAR10_training_data'
    test_file_path = './CIFAR10_test_data'
    # Definition of classes
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
else:
    raise RuntimeError('Dose not exit this kind of data')
# ------------------------------------------------------------
# Number of classes
num_of_classes = len(classes)
# ------------------------------------------------------------
[year, mon, day, hour, minu, sec, _, _, _] = time.localtime(time.time())
# ------------------------------------------------------------
# path of the trained model
model_path = model_name+'_'+data_type+'Mon'+str(mon)+'Day'+str(day)+'Hour'+str(hour)+'Minu'+str(minu)+'Sec'+str(sec)+'.pkl'
print('model_path=', model_path)
##############################################################
# Judgement of the GPU for parallel training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Print the ID of GPU
print(device)
##############################################################
# Defination of the transformation for figures
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor()
    ])
##############################################################
# Showing Image


# def imshow(img):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
##############################################################
# Define the neural network


# ResNet152
if model_name == 'ResNet152':
    net = torchvision.models.resnet152(num_classes=num_of_classes)
# ResNet101
elif model_name == 'ResNet101':
    net = torchvision.models.resnet101(num_classes=num_of_classes)
# ResNet50
elif model_name == 'ResNet50':
    net = torchvision.models.resnet50(num_classes=num_of_classes)
# ResNet34
elif model_name == 'ResNet34':
    net = torchvision.models.resnet34(num_classes=num_of_classes)
# ResNet18
elif model_name == 'ResNet18':
    net = torchvision.models.resnet18(num_classes=num_of_classes)
# VGG19
elif model_name == 'VGG19':
    net = torchvision.models.vgg19(num_classes=num_of_classes)
# VGG16
elif model_name == 'VGG16':
    net = torchvision.models.vgg16(num_classes=num_of_classes)
# VGG13
elif model_name == 'VGG13':
    net = torchvision.models.vgg13(num_classes=num_of_classes)
# VGG11
elif model_name == 'VGG11':
    net = torchvision.models.vgg11(num_classes=num_of_classes)
# Alexnet
elif model_name == 'AlexNet':
    net = torchvision.models.alexnet(num_classes=num_of_classes)
# SqueezeNet1_0
elif model_name == 'SqueezeNet1_0':
    net = torchvision.models.squeezenet1_0(num_classes=num_of_classes)
# SqueezeNet1_1
elif model_name == 'SqueezeNet1_1':
    net = torchvision.models.squeezenet1_1(num_classes=num_of_classes)
# DenseNet121
elif model_name == 'DenseNet121':
    net = torchvision.models.densenet121(num_classes=num_of_classes)
# DenseNet169
elif model_name == 'DenseNet169':
    net = torchvision.models.densenet169(num_classes=num_of_classes)
# DenseNet161
elif model_name == 'DenseNet161':
    net = torchvision.models.densenet161(num_classes=num_of_classes)
# DenseNet201
elif model_name == 'DenseNet201':
    net = torchvision.models.densenet201(num_classes=num_of_classes)
# Inception_v3
elif model_name == 'Inception_v3':
    net = torchvision.models.inception_v3(num_classes=num_of_classes)
else:
    raise RuntimeError('Dose not exit this kind of model')
# ------------------------------------------------------------
net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    torch.backends.cudnn.benchmark = True
#####################################################################
#####################################################################
# Definition of the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
optimizer.zero_grad()
###################################################################################
# Define the tramsformation of training and test figures
transform_train = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
###################################################################################
# Load the training and test data
train_loader = datasets.ImageFolder(training_file_path, transform=transform_train)
trainloader = torch.utils.data.DataLoader(train_loader, batch_size=define_batch_size, shuffle=False)
# test_loader = datasets.ImageFolder(test_file_path, transform=transform_test)
# testloader = torch.utils.data.DataLoader(test_loader, batch_size=define_batch_size, shuffle=False)
####################################################################################
# Training the Neural network


for epoch in range(num_iterations):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader, 0):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        # print('INPUT in training is ', inputs)
        # print('HERE!!!!!!!!!!!!!!!!!!!!')
        # print('output=', outputs.size())
        # labels = labels.view(define_batch_size)  #
        labels = labels.type(torch.cuda.LongTensor)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        if i % 20 == 19:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
torch.save(net.state_dict(), model_path)
print('Finished Training')
# net.cpu()
# torch.save(net.state_dict(), model_path)
# net.cuda(args.cuda)
############################################################
