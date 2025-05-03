#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn
from torchvision import transforms
import torchvision
from torch.utils import data


# In[2]:


def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="./data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,num_workers=4),
data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=4))


# In[4]:


batch_size=256
train_iter,test_iter=load_data_fashion_mnist(batch_size)

net=nn.Sequential(nn.Flatten(), nn.Linear(784,10))


# In[5]:


def init_weights(m):
    if type(m)==nn.Linear:
        nn.init.normal_(m.weight,std=0.01)


# In[6]:


net.apply(init_weights)
loss=nn.CrossEntropyLoss(reduction='none')
trainer=torch.optim.SGD(net.parameters(),lr=0.1)
num_epochs=10


# In[7]:


class Accumulator:
    def __init__(self,n):
        self.data=[0.0]*n
    def add(self,*args):
        self.data=[a+float(b) for a, b in zip(self.data,args)]
    def reset(self):
        self.data=[0.0]*len(self.data)
    def __getitem__(self, item):
        return self.data[item]


# In[8]:


def accuracy(y_hat,y):
    if len(y_hat.shape)>1 and y_hat.shape[1]>1:
        y_hat=y_hat.argmax(axis=1)
    cmp=y_hat.type(y.dtype)==y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net,data_iter):
    net.eval()
    metric=Accumulator(2)
    with torch.no_grad():
        for X,y in data_iter:
            metric.add(accuracy(net(X),y),y.numel())
    return metric[0]/metric[1]

def train_epoch(net, train_iter, loss, updater):
    net.train()
    metric = Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        # print(y_hat)
        l = loss(y_hat, y)
        updater.zero_grad()
        l.mean().backward()
        updater.step()
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    for epoch in range(num_epochs):
        train_metrics = train_epoch(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        print(f'epoch {epoch + 1}, train_metrics {train_metrics}, test_acc {test_acc}')


# In[9]:


train_ch3(net,train_iter,test_iter,loss,num_epochs,trainer)


# In[10]:


def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def predict(net,test_iter):
    for X,y in test_iter:
        break
    preds=get_fashion_mnist_labels(net(X).argmax(axis=1))
    return preds


# In[11]:


predict(net,test_iter)


# In[ ]:




