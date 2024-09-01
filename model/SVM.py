import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error
import torch
import pandas as pd
from tqdm import tqdm
import time
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

df = pd.read_csv("./output_data.csv", index_col=None, header=None)
data = df.values
reshaped_data = data.ravel(order='F')

reshaped_data = np.array(reshaped_data)
newdata = reshaped_data.reshape((365, 20, 288)).transpose((0, 2, 1))

insert_positions = [1, 2, 3, 6, 7, 9, 11, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 25, 28, 29]
expanded_x = np.zeros((365, 288, 30))
expanded_x[:, :, insert_positions] = newdata

edgeindex = [[0, 1], [0, 2], [1, 3], [2, 3], [1, 4], [1, 5], [3, 5], [4, 6], [5, 6], [5, 27], [5, 7], [7, 27],
             [27, 26], [26, 29], [29, 28], [26, 28], [26, 24], [24, 25], [5, 8], [8, 10], [8, 9], [5, 9], [9, 20],
             [20, 21], [9, 16], [15, 16], [3, 11], [11, 12], [11, 17], [11, 15], [17, 18], [18, 19], [9, 19], [9, 23],
             [11, 13], [13, 14], [14, 22], [22, 23], [23, 24]]
edgeindex = np.array(edgeindex)
edgeindex = edgeindex.transpose()
# print(edgeindex.shape)

newy = pd.read_csv("./all_a_array_2.csv", index_col=None,header=None)
newy= newy.values
# newy = newy.ravel(order='F')
newy = newy.reshape((365,288,6))

expanded_x = torch.from_numpy(expanded_x).float()
newy = torch.from_numpy(newy).float()

edgeindex = torch.from_numpy(edgeindex).long()

# 假设 expanded_x 是您的特征张量，newy 是您的标签张量
# 将数据从三维形状转换为二维形状（样本 x 特征）
X = expanded_x.reshape(-1, 30)  # 展平成二维数组，每个样本30个特征
y = newy.reshape(-1, 6)  # 展平成二维数组，每个样本6个输出

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 将数据分割为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 使用径向基函数（RBF）核设置SVM
svm_model = SVR(kernel='rbf')

# 由于SVM默认不支持多输出回归，我们使用MultiOutputRegressor来处理多输出
multioutput_svm = MultiOutputRegressor(svm_model)

print("Training model...")
# 训练模型
multioutput_svm.fit(X_train, y_train)
print("Training complete.")
# 在测试集上进行预测
print("Predicting...")
start_time = time.time()
y_pred = np.array([multioutput_svm.predict(X_test[i:i+1]) for i in tqdm(range(len(X_test)))]).squeeze()
end_time = time.time()  # 记录预测结束时间
prediction_time = end_time - start_time  # 计算预测耗时 1.8s/天
print(f"Prediction time: {prediction_time} seconds")
# 计算均方误差用于评估
mse = mean_squared_error(y_test, y_pred)#7.3
print(f"Mean Squared Error: {mse}")
mae= mean_absolute_error(y_test, y_pred)#1.32
print(f"Mean Absolute Error: {mae}")
