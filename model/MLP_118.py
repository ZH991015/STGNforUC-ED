import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from matplotlib import pyplot as plt


class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        # Flatten the data: (batch_size, seq_length, input_size) -> (batch_size * seq_length, input_size)
        batch_size, seq_length, input_size = x.size()
        x = x.view(batch_size * seq_length, input_size)
        out = self.network(x)
        # Reshape back to sequence data format: (batch_size * seq_length, output_size) -> (batch_size, seq_length, output_size)
        out = out.view(batch_size, seq_length, output_size)
        return out


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 数据加载和预处理
df = pd.read_csv("./all_busloads.csv", index_col=None, header=None)
data = df.values
reshaped_data = data.ravel(order='F')
reshaped_data = np.array(reshaped_data)
newdata = reshaped_data.reshape((366, 99, 288)).transpose((0, 2, 1))

# 插入零点
zeros = np.zeros((newdata.shape[0], newdata.shape[1], 1))
zero_point = [4, 8, 9, 24, 25, 29, 36, 37, 60, 62, 63, 64, 67, 68, 70, 80, 86, 88, 110]
for i in zero_point:
    newdata = np.concatenate((newdata[:, :, :i], zeros, newdata[:, :, i:]), axis=2)

expanded_x = newdata

newy = pd.read_csv("./y1.csv", index_col=None, header=None)
newy = newy.values
newy = newy.ravel(order='F')
newy = np.array(newy)
newy = newy.reshape((366, 288, 54))

expanded_x = torch.from_numpy(expanded_x).float()
newy = torch.from_numpy(newy).float()
output_size = 54
# 定义模型
model = MLPModel(input_size=118, hidden_size=256, output_size=54).to(device)

# 损失函数和优化器
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = CosineAnnealingLR(optimizer, T_max=200, eta_min=0)

# 数据加载器
dataset = TensorDataset(expanded_x, newy)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


def train(model, data_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)


def test(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    total_loss2 = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            mse_losses=torch.nn.functional.mse_loss(outputs, labels)
            total_loss2+= mse_losses.item()
            total_loss += loss.item()
    return total_loss / len(data_loader),total_loss2/len(data_loader)


# 训练和测试循环
train_losses = []
test_losses = []
for epoch in range(200):
    train_loss = train(model, train_loader, criterion, optimizer)
    start_time=time.time()
    test_loss,mse_test_loss = test(model, test_loader, criterion)
    end_time=time.time()
    print(f"time: {end_time-start_time}")
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    scheduler.step()
    print(f'Epoch {epoch + 1}, Training Loss: {train_loss}, Test Loss: {test_loss},MSE Test Loss: {mse_test_loss}')

# 绘制损失曲线
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
