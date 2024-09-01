import time
import numpy as np
from torch import nn
import torch.optim as optim
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR
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
window_size = 12
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
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_sizes = hidden_sizes
        self.num_layers = len(hidden_sizes)
        self.output_size = output_size

        # 创建多层LSTM
        self.lstms = nn.ModuleList([
            nn.LSTM(input_size if i == 0 else hidden_sizes[i - 1], hidden_size, 1, batch_first=True)
            for i, hidden_size in enumerate(hidden_sizes)
        ])
        self.fc1 = nn.Linear(hidden_sizes[-1], 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc = nn.Linear(64, output_size)

    def forward(self, x):
        # 初始化隐藏状态和单元状态
        hs = [torch.zeros(1, x.size(0), hs).to(x.device) for hs in self.hidden_sizes]
        cs = [torch.zeros(1, x.size(0), hs).to(x.device) for hs in self.hidden_sizes]

        # 逐层传递LSTM的输出到下一层
        for i, lstm in enumerate(self.lstms):
            x, (hs[i], cs[i]) = lstm(x, (hs[i], cs[i]))

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        out = self.fc(x)
        return out
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
input_size=30
hidden_sizes=[128, 128, 64, 64]
output_size=6
# 创建模型实例
model = LSTMModel(input_size, hidden_sizes, output_size)
model.to(device)

# 定义损失函数和优化器
criterion = nn.L1Loss()
optimizer = optim.Adamax(model.parameters(), lr=0.001)
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0)

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    scheduler.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")
    # 在验证集上评估模型
    model.eval()
    val_loss = 0.0
    total_loss=0.0
    start_time=time.time()
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()
            total_loss+=nn.functional.mse_loss(outputs, labels).item()
    end_time = time.time()
    print(f"time:{end_time-start_time}")
    print(f"Validation Loss: {val_loss / len(val_loader)},Total Loss: {total_loss / len(val_loader)}")