import numpy as np
from torch import nn
import torch.optim as optim
import os
from torch_geometric.loader import DataLoader
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GraphNorm
from torch.nn import Conv1d, Linear, Sequential, ReLU, BatchNorm1d
from torch_geometric.data import Data
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
from model1 import GTCN_TCN
import sympy as sp
import time
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
window_size=12
# 创建模型实例
model = GTCN_TCN(num_node_features=window_size,num_nodes=118,channel=512).to(device)

checkpoint = torch.load('./best_model_MAE/best_model_epoch_395_loss_0.0786.pth')

# 从checkpoint中提取模型状态字典
model_state_dict = checkpoint['model_state_dict']
# 加载模型权重
model.load_state_dict(model_state_dict)
# 设置为
model.train()

# 数据准备
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
criterion = torch.nn.L1Loss()

df_edges = pd.read_excel('./line.xlsx', header=None)
#print(df_edges)
# 将DataFrame转换为边的列表形式
edgeindex = df_edges.values.tolist()
edgeindex = np.array(edgeindex)
edgeindex = edgeindex.transpose()-1

newy = pd.read_csv("./y1.csv", index_col=None, header=None)
newy= newy.values
newy = newy.ravel(order='F')
newy=np.array(newy)
newy = newy.reshape((366,288,54))

expanded_x = torch.from_numpy(expanded_x).float()
newy = torch.from_numpy(newy).float()
edgeindex = torch.from_numpy(edgeindex).long()
step_size =1 # 步长，控制滑动窗口的移动步长，例如，24代表每次移动24个时段
data_list = []  # 新的数据列表，用于存储所有切割后的数据

for i in range(366):
    x = expanded_x[i, :, :].t()  # 原始的30*288特征矩阵
    y = newy[i, :, :].t()  # 原始的6*288标签矩阵
    edge_index = edgeindex  # 邻接矩阵不变

    # 对一天内的数据使用滑动窗口切割
    for start in range(0, 288 - window_size + 1, step_size):
        end = start + window_size
        x_window = x[:, start:end]
        y_window = y[:, start:end]
        # 创建新的Data对象
        data_window = Data(x=x_window, edge_index=edge_index, y=y_window)
        data_list.append(data_window)
class MyDataset(nn.Module):
    def __init__(self, data_list):
        super(MyDataset, self).__init__()
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

dataset = MyDataset(data_list)
print(len(data_list))
dataset_size = len(data_list)
train_size = int(dataset_size * 0.95)
val_size = dataset_size - train_size
train_dataset=dataset[:train_size]
val_dataset=dataset[train_size:]
# print(len(val_dataset))
# train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
bitch_size=32
# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=bitch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=bitch_size, shuffle=True)
class MyDataset(nn.Module):
    def __init__(self, data_list):
        super(MyDataset,self).__init__()
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]
dataset = MyDataset(data_list)
all_predictions = []  # 用于保存验证过程中的所有预测值
all_targets = []      # 用于保存验证过程中的所有真实值
# 定义惩罚项的系数
sigma1 = torch.tensor(0.5)
sigma2 = torch.tensor(0.5)
# 定义修改后的损失函数
def custom_loss(outputs, targets):
    L1= criterion(outputs, targets)
    # 计算特征和与标签和之间的差距的绝对值作为惩罚项
    feature_sum = outputs.sum(dim=1)
    target_sum = targets.sum(dim=1)
    L2 = torch.abs(feature_sum - target_sum).mean()
    return (1/sigma1**2)*L1 + (1/sigma2**2) * L2+2*torch.log(sigma1) + 2*torch.log(sigma2)

optimizer = torch.optim.Adam(model.parameters(), lr=0.00001,weight_decay=0.0005)
scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=0)
# 迭代数据加载器
best_val_loss =0.08
best_epoch = 0  # 跟踪最佳epoch
save_path = './best_model'  # 设置保存最佳模型的路径
def save_model(epoch, model, optimizer, loss, save_path='./models'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_filename = os.path.join(save_path, f'best_model_epoch_{epoch}_loss_{loss:.4f}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, save_filename)
    print(f'Best model saved to {save_filename} at epoch {epoch} with loss: {loss:.4f}')
# Training loop

def train():
    model.train()
    total_loss = 0
    # all_predictions = []  # Used to save predictions during training
    # all_targets = []      # Used to save true targets during training

    for step, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        y = data.y.view(-1, window_size*54)
        loss = custom_loss(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Save predictions and true targets for calculating R2 score
        # all_predictions.extend(out.detach().cpu().numpy().tolist())
        # all_targets.extend(y.detach().cpu().numpy().tolist())

    # Calculate R2 score for training set
    # r2_train = r2_score(all_targets, all_predictions)

    return total_loss / len(train_loader)

PREDICTION_THRESHOLD = 1 #
# Validation function
def validate():
    model.eval()
    total_loss = 0
    all_predictions = []  # 用于保存验证过程中的所有预测值
    all_targets = []      # 用于保存验证过程中的所有真实值

    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            out = model(data)
            out= torch.where(abs(out) < PREDICTION_THRESHOLD, torch.zeros_like(out), out)
            y = data.y.view(-1, window_size*54)
            loss = criterion(out, y)
            total_loss += loss.item()
            all_predictions.extend(out.cpu().numpy().tolist())  # 保存预测值
            all_targets.extend(y.cpu().numpy().tolist())       # 保存真实值
        # 计算R2 Score
        r2 = r2_score(all_targets, all_predictions)
    return total_loss / len(val_loader), all_predictions, all_targets,r2

# Training and Validation
train_losses = []  # 用于记录每个epoch的训练损失
val_losses = []    # 用于记录每个epoch的验证损失
counter=0
patience = 1000
min_delta = 0.001
for epoch in range(300):  # number of epochs
    train_loss = train()
    val_loss, epoch_val_preds, epoch_val_targets,r2 = validate()
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    # if epoch <= 50:
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    if val_loss < best_val_loss:  # 检查这个epoch的验证损失是否是目前为止最低的
        counter = 0
        best_epoch = epoch  # 更新最佳epoch
        # 仅在当前epoch的验证损失为最低时保存模型
        save_model(epoch, model, optimizer, val_loss, save_path)
        best_val_loss = val_loss  # 更新最佳验证损失
        val_predictions = []
        val_targets = []
        val_predictions.extend(epoch_val_preds)
        val_targets.extend(epoch_val_targets)
        # 将预测值和真实值转换为DataFrame
        val_df = pd.DataFrame({
            'Val_Predictions': [item for sublist in val_predictions for item in sublist],
            'Val_Targets': [item for sublist in val_targets for item in sublist]
        })
        # 保存到CSV文件
        val_df.to_csv('./prediction.csv', index=False)
        print("验证预测值和真实值已保存到CSV文件。")
        print(f'Training complete. Best model was at epoch {best_epoch} with validation loss: {best_val_loss:.4f}')
    else:
        counter += 1
        if counter >= patience:
            print(f'Early stopping triggered. Stopping training at epoch {epoch}.')#早停机制
            break
    print(f"Epoch: {epoch}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, R2 Score: {r2:.4f},Current Learning Rate: {current_lr:.6f}")
# final_model_path = './models/final_model.pth'
# torch.save(model.state_dict(), final_model_path)
# print(f'Final model parameters saved to {final_model_path}')

# 绘制损失曲线
plt.figure(figsize=(10, 6))
plt.plot(train_losses[10:], label='Training Loss')
plt.plot(val_losses[10:], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
from sklearn.metrics import mean_squared_error, mean_absolute_error
# 计算MSE
mse = mean_squared_error(val_targets, val_predictions)
print("Mean Squared Error (MSE):", mse)

# 计算MAE
mae = mean_absolute_error(val_targets, val_predictions)
print("Mean Absolute Error (MAE):", mae)

# 计算RMSE
rmse = np.sqrt(mse)
print("Root Mean Squared Error (RMSE):", rmse)

# 将预测值和真实值转换为DataFrame
val_df = pd.DataFrame({
    'Val_Predictions': [item for sublist in val_predictions for item in sublist],
    'Val_Targets': [item for sublist in val_targets for item in sublist]
})

# 保存到CSV文件
val_df.to_csv('./prediction.csv', index=False)
print("验证预测值和真实值已保存到CSV文件。")