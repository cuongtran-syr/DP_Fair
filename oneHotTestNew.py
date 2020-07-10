import sklearn as skl
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import scale
import numpy as np
import torch
from torch import nn
from torch.utils import data
import argparse

parser = argparse.ArgumentParser(description='Test')

parser.add_argument('--xlen', type=int, default=5)
parser.add_argument('--size', default=10000, type=int)
args = parser.parse_args()


class AvgNet(nn.Module):
    def __init__(self, in_len, unit_len):
        super(AvgNet, self).__init__()

        self._in_len = in_len
        self._unit_len = unit_len
        self._nunits = in_len // unit_len

        self._l1 = nn.Linear(in_len, 1)

    def forward(self, x):
        h = self._l1(x)
        return h


xlen = 5
size = 10000

x = np.array([[np.random.randint(0, 9) for z in range(0, xlen)] for zz in range(0, size)])
y = np.mean(x, 1)
y = y[:, None]

ohe = OneHotEncoder()

x_oh = ohe.fit_transform(x).toarray()

print("x_oh shape: ", x_oh.shape)
print("x_oh = ", x_oh)
print("y = ", y)

# +++
# convert to tensor
training_x = torch.Tensor(x_oh)
training_y = torch.Tensor(y)

# +++
# create dataset
train_tensor = data.TensorDataset(training_x, training_y)

# +++
# pass to data loader
batch_size = 200
train_loader = data.DataLoader(dataset=train_tensor, batch_size=batch_size, shuffle=False)
# length of the train loader is the number of  batches
# ( total set divided by batch size )

num_epochs = 100
unit_len = len(training_x[0]) // xlen
learning_rate = 0.001
model = AvgNet(len(training_x[0]), unit_len)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

loss_list = []  # JK
total_step = len(train_loader)
for epoch in range(num_epochs):
    # +++
    # this is where the loader is used,
    # Note it's accessed by the 'in' keyword
    for i, (images, labels) in enumerate(train_loader):

        # Forward pass
        # +++
        # this is where the DNN (Net3 above) is applied to your data
        outputs = model(images)

        # +++
        # this is where the labels are used (training loss)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())  # JK
        if (i + 1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

print("x[0] = ", x[0:10])
print("y[0] = ", y[0:10])
example_in = torch.Tensor(x_oh[0:10])
print("printing one output:")
print(model(example_in))


