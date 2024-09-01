import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split,Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
from matplotlib import pyplot as plt


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of each time step
        out = self.fc(out)  # Apply the linear layer to the outputs of all time steps
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

# 定义模型
model = LSTMModel(input_size=118, hidden_size=128, output_size=54, num_layers=3, dropout=0.5).to(device)

# 损失函数和优化器
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=200, eta_min=0)

# 数据加载器
# 假设 expanded_x 和 newy 是您的时序数据和标签
dataset = TensorDataset(expanded_x, newy)
num_samples = len(dataset)
train_size = int(0.8 * num_samples)  # 80% of data for training

# 按时间顺序划分训练集和测试集
train_dataset = Subset(dataset, range(train_size))
test_dataset = Subset(dataset, range(train_size, num_samples))

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)  # 时序数据通常不进行洗牌
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

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
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(data_loader)


# 训练和测试循环
train_losses = []
test_losses = []
for epoch in range(200):
    train_loss = train(model, train_loader, criterion, optimizer)
    test_loss = test(model, test_loader, criterion)
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    scheduler.step()
    print(f'Epoch {epoch + 1}, Training Loss: {train_loss}, Test Loss: {test_loss}')

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
