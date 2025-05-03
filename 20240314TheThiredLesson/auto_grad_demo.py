import torch

x = torch.randn(1, 5, requires_grad=True)
y = torch.randn(5, 3, requires_grad=True)
z = torch.randn(3, 1, requires_grad=True)
print("x:\n", x, "\ny\n", y, "\nz\n", z)
xy = torch.matmul(x, y)
print("xy\n", xy)
xyz = torch.matmul(xy, z)
xyz.backward()
print(x.grad, y.grad, z.grad)

zy = torch.matmul(y, z).view(-1, 5)
print(zy, zy.shape, zy[0], zy[0][1])