import os
import numpy as np
import pandas as pd
import torch

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt


# 配置GPU，这里有两种方式
# 方案一：使用os.environ
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# 方案二：使用“device”，后续对要使用GPU的变量用.to(device)即可
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


# 配置其他超参数，如batch_size, num_workers, learning rate, 以及总的epochs
batch_size = 256
num_workers = 0
lr = 1e-4
epochs = 6


def get_train_test_loader():
    # 首先设置数据变换
    image_size = 28
    data_transform = transforms.Compose([
        # transforms.ToPILImage(),   # 这一步取决于后续的数据读取方式，如果使用内置数据集则不需要
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])

    train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=data_transform)
    test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=data_transform)
    classes = train_data.classes
    print("train_data:\n", train_data)
    print("\ntest_data:\n", test_data)

    # 构建训练和测试数据集完成后，需要定义DataLoader类，以便在训练和测试时加载数据
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print("train_loader length {},\n{}".format(len(train_loader), train_loader))
    print("\ntest_loader length {},\n{}".format(len(test_loader), test_loader))
    return train_loader, test_loader, classes


# def show_one_sample(train_loader):
#     image, label = next(iter(train_loader))
#     print(image.shape, label.shape)
#     print(label)
#     plt.imshow(image[0][0], cmap="gray")
#     plt.savefig("image[0][0].png")

def show_one_sample(train_loader,classes):
    image,label = next(iter(train_loader))
    print(image.shape,label.shape)
    print(classes[label[0]])
    plt.imshow(image[0][0],cmap="gray")
    plt.savefig(classes[label[0]]+".png")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3),
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 64 * 4 * 4)
        x = self.fc(x)
        # x = nn.functional.normalize(x)
        return x


class TrainningValidation():

    def __init__(self, model, train_loader, test_loader, optimizer, criterion):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion

    def train(self, epoch):
        self.model.train()
        train_loss = 0
        for data, label in self.train_loader:
            # data, label = data.cuda(), label.cuda()
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, label)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()*data.size(0)
        train_loss = train_loss/len(self.train_loader.dataset)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
        return True

    def val(self, epoch):
        self.model.eval()
        val_loss = 0
        gt_labels = []
        pred_labels = []
        with torch.no_grad():
            for data, label in self.test_loader:
                # data, label = data.cuda, label.cuda
                output = self.model(data)
                preds = torch.argmax(output, 1)
                gt_labels.append(label.cpu().data.numpy())
                pred_labels.append(preds.cpu().data.numpy())
                loss = self.criterion(output, label)
                val_loss += loss.item()*data.size(0)
            val_loss = val_loss/len(self.test_loader.dataset)
            gt_labels, pred_labels = np.concatenate(gt_labels), np.concatenate(pred_labels)
            acc = np.sum(gt_labels == pred_labels)/len(pred_labels)
            print('Epoch: {} \tValidation Loss: {:.6f}, Accuracy: {:6f}\n'.format(epoch, val_loss, acc))
            return True

    def train_val_epochs(self, epochs):
        for epoch in range(1, epochs + 1):
            self.train(epoch=epoch)
            self.val(epoch=epoch)

        self.save_model()

    def save_model(self, model_path=None):
        if model_path is None:
            model_path = "./FahionModel.pkl"
        torch.save(self.model, model_path)
        return True


if __name__ == '__main__':
    train_loader, test_loader ,classes= get_train_test_loader()
    show_one_sample(train_loader,classes)

    model = Net()
    # model = model.cuda()
    # 多卡训练时的写法，之后的课程中会进一步讲解
    # model = nn.DataParallel(model).cuda()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    criterion = nn.CrossEntropyLoss()
    print("nn.CrossEntropyLoss: ", criterion)
    # criterion = nn.CrossEntropyLoss(weight=torch.tensor([1,1,1,1,3,1,1,1,1,1]))
    # print("nn.CrossEntropyLoss with weight: ", criterion)

    tv = TrainningValidation(model=model, train_loader=train_loader, test_loader=test_loader,
                    optimizer=optimizer, criterion=criterion)
    tv.train_val_epochs(epochs=epochs)
