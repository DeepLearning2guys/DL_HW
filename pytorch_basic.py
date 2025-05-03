import torch
import torch.nn.functional as F
import torchvision as tv
import cv2 as cv
import numpy as np
from torch.utils.data import DataLoader


transform = tv.transforms.Compose([tv.transforms.ToTensor(),
                                   tv.transforms.Normalize((0.5,), (0.5,)),
                              ])

train_ts = tv.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_ts = tv.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_dl = DataLoader(train_ts, batch_size=32, shuffle=True, drop_last=False)
test_dl = DataLoader(test_ts, batch_size=64, shuffle=True, drop_last=False)
index = 0

for i_batch, sample_batched in enumerate(train_dl):
    print(i_batch, sample_batched[0].size(), sample_batched[1].size())
    image = sample_batched[0][0].numpy().reshape((28, 28))
    print(image.shape)
    cv.imshow("digit-image", image)
    cv.waitKey(0)
    if index == 4:
        break
    index += 1


m = torch.nn.LogSoftmax(dim=1)
loss = torch.nn.NLLLoss()
# input is of size N x C = 2 X 2
input = torch.randn(2, 2, requires_grad=True)
# each element in target has to have 0 <= value < C
target = torch.tensor([0, 1])
output = loss(m(input), target)
print("m(input)", m(input))
print(target)
print(output)


def image_blur():
    image = cv.imread("D:/images/1024.png", cv.IMREAD_GRAYSCALE)
    h, w = image.shape
    print(h, w)
    cv.imshow("input", image)
    img = np.reshape(image, (1, 1, h, w))
    img = np.float32(img)
    k = torch.ones((1, 1, 7, 7), dtype=torch.float) / 49.0
    z = F.conv2d(torch.from_numpy(img), k, padding=3)
    result = z.numpy()
    print(result.shape)
    result = np.reshape(result, (h, w))
    cv.imshow("result-pytorch-conv2d-demo", np.uint8(result))
    cv.waitKey(0)
    cv.destroyAllWindows()


