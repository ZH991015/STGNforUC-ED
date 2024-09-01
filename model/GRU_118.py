import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from matplotlib import pyplot as plt


def create_sliding_windows(data, window_size):
    """
    Creates sliding windows for a batch of time series data.
    Args:
    - data: Input data for a single day, shape [288, features]
    - window_size: Size of the sliding window

    Returns:
    - windows: Array of input windows
    - labels: Corresponding labels for each window
    """
    num_steps, num_features = data.shape
    windows = []
    labels = []

    # Create windows and labels
    for start in range(num_steps - window_size):
        end = start + window_size
        windows.append(data[start:end, :])  # Create the window
        if end < num_steps:
            labels.append(data[end, :])  # Use the next step as the label

    return np.array(windows), np.array(labels)


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, int(hidden_size * 0.75))
        self.fc2 = nn.Linear(int(hidden_size * 0.75), int(hidden_size * 0.5))
        self.fc3 = nn.Linear(int(hidden_size * 0.5), output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = out[:, -1, :]
        out = nn.ReLU()(self.fc1(out))
        out = nn.ReLU()(self.fc2(out))
        out = self.fc3(out)
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
newy = pd.read_csv("./y1.csv", index_col=None, header=None).values
newy = newy.ravel(order='F').reshape((366, 288, 54))

window_size = 24  # 窗口大小，你可以根据模型的需要调整此参数

all_windows = []
all_labels = []

# 处理每一天的数据
for day in range(366):
    day_data = expanded_x[day]  # 提取一天的数据
    windows, labels = create_sliding_windows(day_data, window_size)
    all_windows.extend(windows)
    all_labels.extend(labels)

# 将列表转换为numpy数组
all_windows = np.vstack(all_windows)  # 形状将是 [num_samples, window_size, features]
all_labels = np.vstack(all_labels)    # 形状将是 [num_samples, features]

# 转换为torch张量
all_windows = torch.from_numpy(all_windows).float()
all_labels = torch.from_numpy(all_labels).float()

print("Number of windows:", len(all_windows))
print("Number of labels:", len(all_labels))

# 创建数据加载器
dataset = TensorDataset(all_windows, all_labels)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Initialize model, loss, optimizer, scheduler
model = GRUModel(input_size=118, hidden_size=128, output_size=54, num_layers=3, dropout=0).to(device)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.006)
scheduler = CosineAnnealingLR(optimizer, T_max=200, eta_min=0)

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
            mse_loss = torch.nn.functional.mse_loss(outputs, labels)
            total_loss += loss.item()
            total_loss2 += mse_loss.item()
    return total_loss / len(data_loader), total_loss2 / len(data_loader)

# Training and testing loop
train_losses = []
test_losses = []


for epoch in range(200):
    train_loss = train(model, train_loader, criterion, optimizer)
    test_loss, mse_test_loss = test(model, test_loader, criterion)
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    scheduler.step()
    print(f'Epoch {epoch + 1}, Training Loss: {train_loss}, Test Loss: {test_loss}, MSE Test Loss: {mse_test_loss}')

# Plot loss curves
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
