import numpy as np
from matplotlib import pyplot as plt
from torch import nn
from sklearn.metrics import mean_squared_error,mean_absolute_error
import torch.optim as optim
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import pandas as pd
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
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
expanded_x=newdata

newy = pd.read_csv("./y1.csv", index_col=None, header=None)
newy= newy.values
newy = newy.ravel(order='F')
newy=np.array(newy)
newy = newy.reshape((366,288,54))
# print(edgeindex.shape)

df_edges = pd.read_excel('./line.xlsx', header=None)
#print(df_edges)
# 将DataFrame转换为边的列表形式
edgeindex = df_edges.values.tolist()
edgeindex = np.array(edgeindex)
edgeindex = edgeindex.transpose()-1

expanded_x = torch.from_numpy(expanded_x).float()
newy = torch.from_numpy(newy).float()

edgeindex = torch.from_numpy(edgeindex).long()

from torch_geometric.data import Data
data_list = []
for i in range(366):
    x = expanded_x[i, :, :].t()  # 你的288*30的节点特征矩阵
    y = newy[i]
    edge_index = edgeindex  # 邻接矩阵

    data = Data(x=x, edge_index=edge_index, y=y,)
    data_list.append(data)
# print(data_list[0])
from torch_geometric.data import Dataset
class MyDataset(nn.Module):
    def __init__(self, data_list):
        super(MyDataset, self).__init__()
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


dataset = MyDataset(data_list)
# print(dataset[0])

# 划分数据集
dataset_size = len(dataset)
train_size = int(dataset_size * 0.8)
val_size = dataset_size - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)


class GNN(torch.nn.Module):
    def __init__(self, num_node_features):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(num_node_features,128)
        self.conv2 = GCNConv(128, 64)
        self.fc=torch.nn.Linear(64*118,15552)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5,training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x=torch.flatten(x,start_dim=1)
        x=x.view(-1,64*118)
        x=self.fc(x)
        return x

# Assuming the number of classes is the size of your y's second dimension
model = GNN(num_node_features=288).to(device)

# Define loss function and optimizer
criterion = torch.nn.L1Loss()  # Use MSELoss for regression tasks
# 定义优化器为SGD
optimizer = optim.Adam(model.parameters(), lr=0.0003,weight_decay=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=200, eta_min=0)
train_losses = []  # 用于记录每个epoch的训练损失
val_losses = []    # 用于记录每个epoch的验证损失

# Training loop
def train():
    model.train()
    total_loss = 0
    for step, data in enumerate(train_loader):
        data=data.to(device)
        optimizer.zero_grad()
        out = model(data)
        y = data.y.view(-1, 15552)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        scheduler.step()
    return total_loss / len(train_loader)

# Validation function
def validate():
    model.eval()
    total_loss = 0
    total_loss2=0
    all_predictions = []  # 用于保存验证过程中的所有预测值
    all_targets = []      # 用于保存验证过程中的所有真实值

    with torch.no_grad():
        start_time = time.time()
        for data in val_loader:
            data = data.to(device)
            out = model(data)
            y = data.y.view(-1, 15552)
            loss = criterion(out, y)
            total_loss += loss.item()
            mse_loss=F.mse_loss(out, y)
            total_loss2+=mse_loss.item()
            all_predictions.extend(out.cpu().numpy().tolist())  # 保存预测值
            all_targets.extend(y.cpu().numpy().tolist())       # 保存真实值

        end_time=time.time()
        print("Validation time: {:.2f}s".format(end_time - start_time))
    return total_loss / len(val_loader), all_predictions, all_targets,total_loss2/len(val_loader)
# Training and Validation
val_predictions = []
val_targets = []
for epoch in range(200):  # number of epochs
    train_loss = train()
    val_loss, epoch_val_preds, epoch_val_targets,val_MSEloss = validate()
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_predictions.extend(epoch_val_preds)
    val_targets.extend(epoch_val_targets)

    print(f'Epoch: {epoch}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f},Validation MSE Loss: {val_MSEloss:.4f}')

# 绘制损失曲线
plt.figure(figsize=(10, 6))
plt.plot(train_losses[:], label='Training Loss')
plt.plot(val_losses[:], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
#mae 2.4