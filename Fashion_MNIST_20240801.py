import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader

batch_size=256
epoch=6

def getdata():
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize(mean=[0.5],std=[0.5])])
    
    train_data = datasets.FashionMNIST(root="./data",train=True,transform=transform,download=True)
    test_data = datasets.FashionMNIST(root="./data",train=False,transform=transform,download=True)
    classes = train_data.classes
    print("data is ready")

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size,shuffle=True,drop_last=True)
    test_loader = DataLoader(dataset=test_data,batch_size=batch_size,shuffle=False,drop_last=True)
    print("dataloader is ready")
    return train_loader, test_loader, classes


def get_first_img(train_loader,classes):
    img,label = next(iter(train_loader))
    plt.imshow(img[0][0] ,cmap="binary")
    print(classes[label[0]])
    os.makedirs("./img",exist_ok=True)
    plt.savefig('./img/{}.png'.format(classes[label[0]]))

if __name__ == "__main__":

    train_loader,test_loader,classes = getdata()
    get_first_img(train_loader,classes)

    model = Net()

    optimizer = optim.Adam(model.parameters(),lr=0.001)

    criterion = nn.CrossEntropyLoss

    tv = TrainningValidation(model=model,train_loader=train_loader,test_loader=test_loader,optimizer=optimizer,criterion=criterion)
    tv.train_val_epoches(epoch)