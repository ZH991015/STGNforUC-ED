import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset,TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from matplotlib import pyplot as plt
from torch.nn.utils import weight_norm
from torch.utils.data import random_split
import time
import torch.nn.functional as F
# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
df = pd.read_csv("./output_data.csv", index_col=None, header=None)
data = df.values
reshaped_data = data.ravel(order='F')

reshaped_data = np.array(reshaped_data)
newdata = reshaped_data.reshape((365, 20, 288)).transpose((0, 2, 1))

insert_positions = [1, 2, 3, 6, 7, 9, 11, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 25, 28, 29]
expanded_x = np.zeros((365, 288, 30))
expanded_x[:, :, insert_positions] = newdata

newy = pd.read_csv("./all_a_array_2.csv", index_col=None,header=None)
newy= newy.values
# newy = newy.ravel(order='F')
newy = newy.reshape((365,288,6))

expanded_x = torch.from_numpy(expanded_x).float()
newy = torch.from_numpy(newy).float()
expanded_x = expanded_x.reshape(365 * 288, 30)
newy = newy.reshape(365 * 288, 6)
window_size = 48
step_size = 1
data = []
for start in range(0, 365 * 288 - window_size + 1, step_size):
    end = start + window_size
    x_window = expanded_x[start:end, :]
    y_window = newy[start:end, :]
    data.append((x_window, y_window))

# 将data转换为tensor列表
x_data, y_data = zip(*data)
x_tensor = torch.stack(x_data)
y_tensor = torch.stack(y_data)

# 如果需要将数据加载到GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x_tensor = x_tensor.to(device)
y_tensor = y_tensor.to(device)

print(x_tensor.shape, y_tensor.shape)
# 假设 x_tensor 和 y_tensor 是您的时序数据和标签
num_samples = x_tensor.size(0)
train_size = int(num_samples * 0.8)

# 按顺序划分训练集和验证集
x_train = x_tensor[:train_size]
y_train = y_tensor[:train_size]
x_val = x_tensor[train_size:]
y_val = y_tensor[train_size:]

# 创建数据加载器
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)  # 时序数据通常不进行洗牌
val_dataset = TensorDataset(x_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
# 模型参数
print(x_tensor.shape, y_tensor.shape)
class Chomp1d(nn.Module):
    """ Removes the last elements of a time series to maintain causality in the convolution. """
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    """ A block in the TCN comprising dilated causal convolutions, batch norm, and a residual connection. """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    """ The overall TCN model comprising multiple TemporalBlocks. """
    def __init__(self, input_size, output_size, num_channels, kernel_size=2, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size)]

        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        x = x.transpose(1, 2)  # Convert from (batch_size, seq_length, num_features) to (batch_size, num_features, seq_length)
        x = self.network(x)
        x = x.transpose(1, 2)  # Back to (batch_size, seq_length, num_channels)
        x = self.linear(x)
        return x
model = TCN(input_size=30, output_size=6, num_channels=[64, 128,64,32]).to(device)

criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = CosineAnnealingLR(optimizer, T_max=100,eta_min=0)


train_losses=[]
test_losses=[]
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
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    total_loss2=0
    strat_time=time.time()
    with torch.no_grad():  # No gradients needed
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            mse_loss = torch.nn.functional.mse_loss(outputs, labels)
            total_loss2 += mse_loss.item()
            total_loss += loss.item()
    end_time=time.time()
    print("time:",end_time-strat_time)
    return total_loss / len(data_loader),total_loss2/len(data_loader)
for epoch in range(100):
    train_loss = train(model, train_loader, criterion, optimizer)
    test_loss,mse_test_losses=test(model, val_loader, criterion)
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    scheduler.step()
    print(f'Epoch {epoch+1}, Loss: {train_loss},test Loss:{test_loss},mse_test_loss:{mse_test_losses}')
# 绘制损失曲线
plt.figure(figsize=(10, 6))
plt.plot(train_losses[10:], label='Training Loss')
plt.plot(test_losses[10:], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()#MAE 0.4