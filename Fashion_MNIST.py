import torch
import torchvision
from torchvision import datasets
from tqdm import tqdm
import matplotlib
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os

batch_size=256
epochs = 6

#preprocess the data
def getdata():
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize(mean=[0.5],std=[0.5])])
    
    train_data = datasets.FashionMNIST("data",train=True,download=True,transform=transform)
    print("train data is ready")
    test_data = datasets.FashionMNIST("data",train=False,download=True,transform=transform)
    print("test data is ready")
    classes = train_data.classes
    print("class data is ready")

    train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True,num_workers=0,drop_last=True)
    print("train loader is ready")
    test_loader = DataLoader(test_data,batch_size=batch_size)
    print("test loader is ready")
    return train_loader,test_loader,classes

def show_one_sample(train_loader,classes):
    image,label = next(iter(train_loader))
    print(image.shape,label.shape)
    print(classes[label[0]])
    plt.imshow(image[0][0],cmap="gray")
    # plt.savefig(classes[label[0]]+".png")
    # download_path = os.path.join(args.output, "download")#使用os.path.join函数来拼接路径，创建一个用于下载数据的目录路径。args.output可能是一个命令行参数，指定了数据的根目录
    os.makedirs("./img",exist_ok=True)
    plt.savefig('./img/{}.png'.format(classes[label[0]]))

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5), #卷积层，输入1个通道，输出32个通道，卷积核长度5*5
            nn.ReLU(),#激活层
            nn.MaxPool2d(2,stride=2), #池化层，池化窗口2*2，步长2
            nn.Dropout(0.3), #Dropout层，丢弃概率0.3
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride = 2),
            nn.Dropout(0.3)
        )
        self.fc = nn.Sequential(
            nn.Linear(64*4*4,512),#全连接层，输入特征64*4*4展平，输出512
            nn.ReLU(),
            nn.Linear(512,10)#全连接层
        )
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 64*4*4) #每个图展平成一维，即batchsize*(64*4*4)
        x = self.fc(x)

        return x

class TrainningValidation():

    def __init__(self,model,train_loader,test_loader,optimizer,criterion):
        self.model=model
        self.train_loader=train_loader
        self.test_loader=test_loader
        self.optimizer=optimizer
        self.criterion=criterion

    def train(self,epoch):
        self.model.train()
        train_loss = 0
        test_acc = 0
        for data,label in self.train_loader:
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output,label)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()*data.size(0)
        train_loss = train_loss / len(self.train_loader.dataset)
        print('Epoch:{} \tTraining Loss:{:.6f}'.format(epoch,train_loss))
        return True

    def val(self,epoch):
        self.model.eval()
        test_loss = 0
        test_acc = 0
        for data,label in self.test_loader:
            output = self.model(data)
            loss = self.criterion(output,label)
            preds = torch.argmax(output,1)
            test_loss +=loss.item()*data.size(0)
            equal_elements = (preds == label)
            sum_equal_elements = sum(equal_elements)
            test_acc = test_acc + sum_equal_elements
        test_loss = test_loss / len(self.test_loader.dataset)
        test_acc = test_acc / len(self.test_loader.dataset)
        print('Epoch:{} \tValidation Loss:{:.6f}, Accuracy:{:.6f}\n'.format(epoch,test_loss,test_acc))
        return True

    def train_val_epochs(self,epochs):
        for epoch in range(1,epochs + 1):
            self.train(epoch=epoch)
            self.val(epoch=epoch)
        self.save_model()

    def save_model(self, model_path=None):
        if model_path is None:
            model_path = "./FahionModel.pkl"
        torch.save(self.model,model_path)
        return True

                

if __name__ == "__main__":
    train_loader,test_loader,classes = getdata()
    show_one_sample(train_loader,classes)

    model = Net()#生成模型

    optimizer = optim.Adam(model.parameters(),lr=0.001)#优化器,学习率0.001

    criterion = nn.CrossEntropyLoss()#损失函数
    print("nn.CrossEntropyLoss: ", criterion)

    tv = TrainningValidation(model=model,train_loader=train_loader,test_loader=test_loader,optimizer=optimizer,criterion=criterion)#训练器
    tv.train_val_epochs(epochs=epochs)#训练

