import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader
import torch

batch_size=256
epoch=6

def getdata():
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize(mean=[0.5],std=[0.5])])
    
    train_data = datasets.FashionMNIST(root="./data",train=True,transform=transform,download=True)
    test_data = datasets.FashionMNIST(root="./data",train=False,transform=transform,download=True)
    classes = train_data.classes
    print("data is ready")

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size,shuffle=True,drop_last=False)
    test_loader = DataLoader(dataset=test_data,batch_size=batch_size,shuffle=False,drop_last=False)
    print("dataloader is ready")
    return train_loader, test_loader, classes


def get_first_img(train_loader,classes):
    img,label = next(iter(train_loader))
    plt.imshow(img[0][0] ,cmap="binary")
    print(classes[label[0]])
    os.makedirs("./img",exist_ok=True)
    plt.savefig('./img/{}.png'.format(classes[label[0]]))

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Dropout(0.3),
            nn.Conv2d(in_channels=32, out_channels=64,kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Dropout(0.3)
        )
        self.conv2 = nn.Sequential(
            nn.Linear(in_features=64*4*4,out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512,out_features=10)
        )
    def forward(self,x):
        x = self.conv1(x)
        x = x.view(-1,64*4*4)
        x = self.conv2(x)

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
        for data,label in self.train_loader:
            self.optimizer.zero_grad()
            res = self.model(data)
            loss = self.criterion(res,label)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()*label.size(0)
        train_loss = train_loss / len(self.train_loader.dataset)
        print('Epoch:{} \tTraining Loss={:.6f}'.format(epoch,train_loss))
        return True
    
    def test(self,epoch):
        self.model.eval()
        test_loss = 0
        acc_num = 0
        for data,label in self.test_loader:
            res = self.model(data)
            loss = self.criterion(res,label)
            test_loss += loss.item()*label.size(0)
            pres = torch.argmax(input=res,dim = 1)
            pres_correct = (pres==label)
            pres_correct_num = sum(pres_correct)
            acc_num += pres_correct_num
        test_loss = test_loss / len(self.test_loader.dataset)
        acc_rate = acc_num / len(self.test_loader.dataset)
        print('Epoch:{} \tTesting Loss={:.6f},Accuracy Rate={:.6f}'.format(epoch,test_loss,acc_rate))
        return True
    
    def train_val_epoches(self,epoch):
        for i in range(1,epoch+1):
            self.train(i)
            self.test(i)
        self.save_model()

    def save_model(self,path=None):
        if path is None:
            path = "./FashionModel.pkl"
        torch.save(obj=self.model,f=path)
        return True






if __name__ == "__main__":

    train_loader,test_loader,classes = getdata()
    get_first_img(train_loader,classes)

    model = Net()

    optimizer = optim.Adam(model.parameters(),lr=0.001)

    criterion = nn.CrossEntropyLoss()

    tv = TrainningValidation(model=model,train_loader=train_loader,test_loader=test_loader,optimizer=optimizer,criterion=criterion)
    tv.train_val_epoches(epoch)