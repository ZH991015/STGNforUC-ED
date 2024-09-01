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

expanded_x = torch.from_numpy(expanded_x).float()
newy = torch.from_numpy(newy).float()

# 展平数据，准备训练
X = expanded_x.reshape(-1, 118)  # 展平成二维数组，每个样本30个特征
y = newy.reshape(-1, 54)  # 展平成二维数组，每个样本6个输出
# 标准化特征（对于SVM来说很重要）
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
mae= mean_absolute_error(y_test, y_pred)#0.08  1.32
print(f"Mean Absolute Error: {mae}")
