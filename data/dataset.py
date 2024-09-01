import numpy as np
from torch import nn
import torch.optim as optim
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
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
window_size = 144
step_size = 1
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