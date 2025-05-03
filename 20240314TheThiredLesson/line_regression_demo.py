import numpy as np
import torch
import matplotlib.pyplot as plt

x = np.array([1,2,0.5,2.5,2.6,3.1], dtype=np.float32).reshape((-1, 1))
y = np.array([3.7,4.6,1.65,5.68,5.98,6.95], dtype=np.float32).reshape(-1, 1)


class LinearRegressionModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out


input_dim = 1
output_dim = 1
model = LinearRegressionModel(input_dim, output_dim)
criterion = torch.nn.MSELoss()

learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(100):
    epoch += 1
    # Convert numpy array to torch Variable
    inputs = torch.from_numpy(x).requires_grad_()
    labels = torch.from_numpy(y)

    # Clear gradients w.r.t. parameters
    optimizer.zero_grad()

    # Forward to get output
    outputs = model(inputs)

    # Calculate Loss
    loss = criterion(outputs, labels)

    # Getting gradients w.r.t. parameters
    loss.backward()

    # Updating parameters
    optimizer.step()

    print('epoch {}, loss {}'.format(epoch, loss.item()))

# Purely inference
predicted_y = model(torch.from_numpy(x).requires_grad_()).data.numpy()
print("标签Y:", y)
print("预测Y:", predicted_y)

# Clear figure
plt.clf()

# Get predictions
predicted = model(torch.from_numpy(x).requires_grad_()).data.numpy()

# Plot true data
plt.plot(x, y, 'go', label='True data', alpha=0.5)

# Plot predictions
plt.plot(x, predicted_y, '--', label='Predictions', alpha=0.5)

# Legend and plot
plt.legend(loc='best')
plt.show()


