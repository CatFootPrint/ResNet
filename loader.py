import os
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms


def default_loader(path):
    return Image.open(path).convert('RGB')

mytransform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor()
    ]
)

class myImageFloder(data.Dataset):
    def __init__(self, root, label, transform=None, target_transform=None, loader=default_loader):
        fh = open(label)
        c = 0
        imgs = []
        class_names = []
        for line in fh.readlines():
            if c == 0:
                class_names = [n.strip() for n in line.rstrip().split('	')]
            else:
                cls = line.split()
                fn = cls.pop(0)
                if os.path.isfile(os.path.join(root, fn)):
                    imgs.append((fn, tuple([float(v) for v in cls])))
            c = c + 1
        self.root = root
        self.imgs = imgs
        self.classes = class_names
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(os.path.join(self.root, fn))
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.Tensor(label)

    def __len__(self):
        return len(self.imgs)

    def getName(self):
        return self.classes


def trainingmyImageFloder():
    dataloader = myImageFloder('./training_data',
                               './training_data/training_data.txt', transform=mytransform)
    #print ('training dataloader.getName', dataloader.getName())

    #for index, (img, label) in enumerate(dataloader):
        #img.show()
        #print('img->', img.size(), ' label->', label)
    return dataloader
def testmyImageFloder():
    dataloader = myImageFloder('./test_data',
                               './test_data/test_data.txt', transform=mytransform)
    #print ('test dataloader.getName', dataloader.getName())

    #for index, (img, label) in enumerate(dataloader):
        #img.show()
        #print('img->', img.size(), ' label->', label)
    return dataloader


# if __name__ == "__main__":
#     training=trainingmyImageFloder()
#     test=testmyImageFloder()
#     img_train, label_train = training[0]
#     img_test, label_test = test[0]
# for img, label in training:
#     print(img.size(), label)
# for img, label in test:
#     print(img.size(), label)
