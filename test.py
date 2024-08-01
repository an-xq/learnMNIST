import enum
from pickletools import optimize
from random import shuffle
from re import X
import jittor as jt
from jittor.dataset.mnist import MNIST
import jittor.transform as trans
import matplotlib.pyplot as plt
from jittor import nn, Module
import numpy as np

batch_size = 64

#训练集加载器
train_loader = MNIST(train=True,transform=trans.Resize(28)).set_attrs(batch_size=batch_size,shuffle=True)
#测试集加载器
val_loader = MNIST(train=False,transform=trans.Resize(28)).set_attrs(batch_size=batch_size,shuffle=False)

#测试图像生成
num = 0
for inputs,targets in val_loader:
    print("inputs.shape", inputs.shape)
    print("target.shape", targets.shape)

    plt.imshow(inputs[num].numpy().transpose(1,2,0))
    plt.show()
    print("target:",targets[num].data[0])
    break

#模型
class Model(Module):
    def __init__(self):
        super(Model,self).__init__()
        self.conv1 = nn.Conv(3,32,3,1)#卷积层1
        self.conv2 = nn.Conv(32,64,3,1)#卷积层2
        self.bn = nn.BatchNorm(64)#归化层

        self.max_pool = nn.Pool(2,2)#池化层
        self.relu = nn.Relu()#非线性激活层
        self.fc1 = nn.Linear(64*12*12,256)#线性全连接1
        self.fc2 = nn.Linear(256,10)#线性全连接2

    def execute(self, x):
        x = self.conv1(x)#卷积层1
        print("conv1.shape", x.shape)
        x = self.relu(x)#非线性激活层
        print("relu1.shape", x.shape)

        x = self.conv2(x)#卷积层2
        print("conv2.shape", x.shape)
        x = self.bn(x)#归化层
        print("bn.shape", x.shape)
        x = self.relu(x)#非线性激活层
        print("relu2.shape", x.shape)

        x = self.max_pool(x)#池化层
        print("max_pool.shape", x.shape)
        x = jt.reshape(x,[x.shape[0],-1])#压缩
        print("reshape.shape", x.shape)
        x = self.fc1(x)#线性全连接1
        print("fc1.shape", x.shape)
        x = self.relu(x)#非线性激活层
        print("relu3.shape", x.shape)
        x = self.fc2(x)#线性全连接2
        print("fc2.shape", x.shape)

        return x

model = Model()

#设置损失函数
loss_function = nn.CrossEntropyLoss()

#设置优化器
learning_rate = 0.1
momentum = 0.9
weight_decay = 1e-4
optimizer = nn.SGD(model.parameters(),learning_rate,momentum,weight_decay)

#训练函数
def train(model,train_loader,loss_function,optimizer,epoch):
    model.train()
    train_losses = list()#Loss容器
    for batch_idx,(inputs,targets) in enumerate(train_loader):
        outputs = model(inputs)
        loss = loss_function(outputs,targets)
        optimizer.step(loss)

        if batch_idx %10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx, len(train_loader),100. * batch_idx / len(train_loader), loss.data[0]))
    return train_losses

#测试函数
def test(model,val_loader,loss_function,epoch):
    model.eval()
    total_correct = 0
    total_num = 0
    for batch_idx,(inputs,targets) in enumerate(val_loader):
        outputs = model(inputs)
        pred = np.argmax(outputs.data, axis=1)                         # 根据 10 个分量，选择最大相似度的为预测的数字值
        correct = np.sum(targets.data==pred)                           # 计算本批次中，正确预测的次数，即数据标签等于预测值的数目
        batch_size = inputs.shape[0]                                   # 计算本批次中，数据的总数目
        acc = correct / batch_size                                     # 计算本批次的正确率
        
        total_correct += correct                                       # 将本批次的正确预测次数记录到总数中
        total_num += batch_size                                        # 将本批次的数据数目记录到总数中
        
        if batch_idx % 10 == 0:                                        # 每十个批次，打印一次测试集上的准确率
            print('Test Epoch: {} [{}/{} ({:.0f}%)]\tAcc: {:.6f}'.format(epoch,batch_idx, len(val_loader),100. * float(batch_idx) / len(val_loader), acc))
    test_acc = total_correct / total_num                               # 计算本纪元的正确率
    print ('Total test acc =', test_acc)              
    return test_acc


# 设置纪元数，并开始训练和测试模型
epochs = 5
train_losses = list()
test_acc = list()
for epoch in range(epochs):
    loss = train(model, train_loader, loss_function, optimizer, epoch) # 训练模型，并返回该纪元的 Loss 列表
    train_losses += loss
    acc = test(model, val_loader, loss_function, epoch)                # 测试模型，并返回该纪元的正确率
    test_acc.append(acc)