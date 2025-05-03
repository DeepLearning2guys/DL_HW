#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torchvision as tv
import cv2 as cv
from torch.utils.data import DataLoader


# In[2]:


transform = tv.transforms.Compose([tv.transforms.ToTensor(),
                                   tv.transforms.Normalize((0.5,), (0.5,)),
                              ])

#另外请自己FashionMNIST，并显示
train_ts = tv.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_ts = tv.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_dl = DataLoader(train_ts, batch_size=32, shuffle=True, drop_last=False)
test_dl = DataLoader(test_ts, batch_size=64, shuffle=True, drop_last=False)
index = 0


# In[3]:


for i_batch, sample_batched in enumerate(train_dl):
    print(i_batch, sample_batched[0].size(), sample_batched[1].size())
    image = sample_batched[0][0].numpy().reshape((28, 28))
    print(image.shape)
    image = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX, cv.CV_32F)
    cv.imshow("My-image", image)
    cv.imwrite("./" + "nub.jpg", image)
    cv.waitKey(0)
    if index == 4:
        break
    index += 1

