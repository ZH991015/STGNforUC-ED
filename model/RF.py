import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 数据加载和预处理
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

# 展平数据，准备训练
X = expanded_x.reshape(-1, 30*288)  # 展平成二维数组，每个样本30个特征
y = newy.reshape(-1, 6*288)  # 展平成二维数组，每个样本6个输出

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 使用随机森林回归器
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# 处理多输出回归问题
multioutput_rf = MultiOutputRegressor(rf_model)

print("Training model...")
# 训练模型
multioutput_rf.fit(X_train, y_train)
print("Training complete.")

# 在测试集上进行预测
print("Predicting...")
start_time = time.time()
y_pred = multioutput_rf.predict(X_test)
end_time = time.time()  # 记录预测结束时间
prediction_time = end_time - start_time  # 计算预测耗时
print(f"Prediction time: {prediction_time} seconds")#2.01s/73天

# 计算均方误差和绝对误差用于评估
mse = mean_squared_error(y_test, y_pred)#8.43
mae = mean_absolute_error(y_test, y_pred)#1.58
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
