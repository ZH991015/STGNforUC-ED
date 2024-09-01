import time
import numpy as np
from torch import nn
import torch.optim as optim
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
df = pd.read_csv("./all_busloads.csv", index_col=None, header=None)
data = df.values
reshaped_data = data.ravel(order='F')
reshaped_data = np.array(reshaped_data)
newdata = reshaped_data.reshape((366, 99, 288)).transpose((0, 2, 1))
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
expanded_x = expanded_x.reshape(366 * 288, 118)
newy = newy.reshape(366 * 288, 54)
window_size = 24
step_size = 12
data = []
for start in range(0, 366 * 288 - window_size + 1, step_size):
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

# 模型定义
# 模型定义
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(GRUModel, self).__init__()
        self.hidden_sizes = hidden_sizes
        self.num_layers = len(hidden_sizes)
        self.output_size = output_size  # 添加这一行来初始化output_size属性

        # 创建多层GRU，每层神经元数递减
        self.grus = nn.ModuleList([
            nn.GRU(input_size if i == 0 else hidden_sizes[i - 1], hidden_size, 1, batch_first=True)
            for i, hidden_size in enumerate(hidden_sizes)
        ])
        self.fc1 = nn.Linear(hidden_sizes[-1], 128)
        self.fc2 = nn.Linear(128, 64)
        # 最后的全连接层
        self.fc = nn.Linear(64, output_size)
        self.l1=nn.LayerNorm(hidden_sizes[-1])
    def forward(self, x):
        # 初始化隐藏状态
        h = [torch.zeros(1, x.size(0), hs).to(x.device) for hs in self.hidden_sizes]

        # 逐层传递GRU的输出到下一层
        for i, gru in enumerate(self.grus):
            x, h[i] = gru(x, h[i])
        # 对每个时间步应用全连接层
        x=self.l1(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        out = self.fc(x)  # 全连接层处理
        return out



input_size = 118
hidden_sizes = [512, 256, 128]  # 逐层减少的隐藏单元数
output_size = 54

# 创建模型实例
model = GRUModel(input_size, hidden_sizes, output_size)

# 将模型加载到GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义损失函数和优化器
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
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
            end_time=time.time()
        print(f"time {end_time-start_time}")
    print(f"Validation Loss: {val_loss / len(val_loader)},Total Loss: {total_loss / len(val_loader)}")
