# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 04:05:02 2018

@author: Kezhong
"""
from torch import nn
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
##############################################################
# Definition of the path of trained model
model_file_path = './VGG11_Cyclic1.pkl'
# Definition of the batch size
define_batch_size = 4
# ============================================================
# load test data


if model_file_path.find('Dog_vs_Cat') >= 0:
    training_file_path = './dogcat'
    test_file_path = './dogcat_test'
    # Definition of classes
    classes = ('Cat', 'Dog')
    num_of_classes = len(classes)
    net = torchvision.models.vgg16(num_classes=num_of_classes)
    net.load_state_dict(torch.load(model_file_path))
elif model_file_path.find('Cyclic') >= 0:
    training_file_path = './data_cyclic/training_cyclic'
    test_file_path = './data_cyclic/test_cyclic'
    # Definition of classes
    classes = ('BPSK', 'QPSK')
    num_of_classes = len(classes)
    net = torchvision.models.vgg16(num_classes=num_of_classes)
elif model_file_path.find('CIFAR10') >= 0:
    training_file_path = './CIFAR10_training_data'
    test_file_path = './CIFAR10_test_data'
    # Definition of classes
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    num_of_classes = len(classes)
    net = torchvision.models.vgg16(num_classes=num_of_classes)
else:
    raise RuntimeError('Dose not exit this kind of data')
#######################################################
# The number of classes
print(classes)
# Define the GPU
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(device)
########################################################
# Defination of parameters
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor()
    ]
)
########################################################
# Load the trained model
# net = torch.load(model_file_path)
# net = torch.load(model_file_path, map_location=lambda storage, loc: storage)

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
##################################################################################
# Load the test data
test_loader = datasets.ImageFolder(test_file_path, transform=transform_test)
testloader = torch.utils.data.DataLoader(test_loader, batch_size=define_batch_size, shuffle=False)
####################################################################################
# Definition of Loss function
criterion = nn.CrossEntropyLoss()
# ==================================================================================
# Test stage
class_correct = list(0. for i in range(num_of_classes))
class_total = list(0. for i in range(num_of_classes))
test_loss = 0
with torch.no_grad():
    for i, data in enumerate(testloader, 0):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        net.to(device)
        outputs = net(images)
        testloss = criterion(outputs, labels)
        test_loss += testloss.item()
        # print('output', outputs.data)
        # print('test_loss', test_loss)
        _, predicted = torch.max(outputs, 1)
        # print('output=', outputs.size())
        # torch.max()返回两个结果，第一个是最大值，第二个是对应的索引值；
        # 第二个参数如果是0代表按列取最大值并返回对应的行索引值，
        # 如果是1代表按行取最大值并返回对应的列索引值。
        print('predicted=', predicted.data)
        print('lables=', labels.data)
        # labels = labels.type(torch.cuda.LongTensor)
        c = (predicted == labels).squeeze()
        # print('c=', c.data)
        for item in range(define_batch_size):
            label = labels[item]
            class_correct[label] += (c[item]).item()
            # print('c[i].item()=', c[i].item())
            # print('class_correct=', class_correct)
            class_total[label] += 1
        print('class_correct=', class_correct)


for i in range(num_of_classes):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))


########################################################################

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(define_batch_size)))

########################################################################


correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Total Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
