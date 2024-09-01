import numpy as np
from torch import nn
import torch.optim as optim
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import time
# 数据预处理
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

print(x_tensor.shape, y_tensor.shape)

# 确定验证集的大小
val_size = int(len(x_tensor) * 0.2)

# 按时间顺序划分训练集和验证集
x_train, x_val = x_tensor[:-val_size], x_tensor[-val_size:]
y_train, y_val = y_tensor[:-val_size], y_tensor[-val_size:]

# 创建数据加载器
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)
val_dataset = TensorDataset(x_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
# 定义Highway网络
class Highway(nn.Module):
    def __init__(self, size, num_layers=1, f=F.relu):
        super(Highway, self).__init__()
        self.num_layers = num_layers
        self.f = f
        self.nonlinear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

    def forward(self, x):
        for layer in range(self.num_layers):
            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)
            gate = torch.sigmoid(self.gate[layer](x))
            x = gate * nonlinear + (1 - gate) * linear
        return x


# 定义LSTNet模型
class LSTNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, cnn_kernel_size, rnn_hidden_dim, output_dim, dropout, time_steps
                 ,highway_layers=1):
        super(LSTNet, self).__init__()

        self.cnn = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=cnn_kernel_size,
                             padding=(cnn_kernel_size - 1) // 2)
        self.gru = nn.GRU(hidden_dim, rnn_hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(rnn_hidden_dim, output_dim)

        self.skip_gru = nn.GRU(hidden_dim, rnn_hidden_dim, batch_first=True)
        self.skip_linear = nn.Linear(rnn_hidden_dim, output_dim)
        self.output_dim = output_dim
        self.time_steps = time_steps

        self.highway = Highway(size=output_dim, num_layers=highway_layers)
        self.ar = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        batch_size, seq_len, num_features = x.size()

        # CNN part
        x = x.permute(0, 2, 1)  # (batch_size, num_features, seq_len) for Conv1d
        c = torch.relu(self.cnn(x))  # (batch_size, hidden_dim, seq_len) due to padding
        c = c.permute(0, 2, 1)  # (batch_size, seq_len, hidden_dim)

        # RNN part
        out, _ = self.gru(c)
        out = self.dropout(out)
        out = self.linear(out)

        # Skip-RNN part
        skip_out = []
        skip_step = seq_len // self.time_steps  # Adjust skip_step to match seq_len
        for i in range(0, seq_len - skip_step + 1, skip_step):
            skip_r = c[:, i:(i + skip_step), :].contiguous()
            skip_out_i, _ = self.skip_gru(skip_r)
            skip_out.append(skip_out_i[:, -1, :])

        if len(skip_out) > 0:
            skip_out = torch.stack(skip_out, dim=1)
            skip_out = self.skip_linear(skip_out)
        else:
            skip_out = torch.zeros((batch_size, self.time_steps, self.output_dim)).to(x.device)
        # Ensure the output dimensions are (batch_size, time_steps, output_dim)
        out = out[:, -self.time_steps:, :]  # Take only the last `time_steps` outputs
        combined_out = out + skip_out

        # Highway Network for AR
        highway_out = self.highway(combined_out)
        # Autoregressive component
        ar_out = x[:, :, -self.time_steps:]  # Last `time_steps` of the original input
        ar_out = ar_out.permute(0, 2, 1)  # (batch_size, time_steps, num_features)
        ar_out = self.ar(ar_out)
        return highway_out + ar_out




# 模型参数
input_dim = 118
hidden_dim = 128
cnn_kernel_size = 3
rnn_hidden_dim = 50
output_dim = 54
dropout = 0.2
time_steps = window_size
highway_layers = 2

# 创建模型实例
model = LSTNet(input_dim, hidden_dim, cnn_kernel_size, rnn_hidden_dim, output_dim, dropout, time_steps, highway_layers)

# 将模型加载到GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义损失函数和优化器
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.003)
scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=0)
# 训练模型
num_epochs = 50
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
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

    model.eval()
    val_loss = 0.0
    total_loss = 0.0
    start_time = time.time()
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            val_loss += criterion(outputs, labels).item()
            total_loss += nn.functional.mse_loss(outputs, labels).item()
    print(f"Validation Loss: {val_loss / len(val_loader)},mse Loss: {total_loss / len(val_loader)}")
    end_time = time.time()
    print(f"Inference time: {end_time - start_time}s")