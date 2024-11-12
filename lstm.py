import math
from matplotlib import pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import StandardScaler

# 定義激活函數及其導數
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

class LSTM:
    def __init__(self, input_size, hidden_size, learning_rate):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        # 遺忘門權重與偏置
        self.w_f = np.random.randn(hidden_size, hidden_size + input_size)
        self.b_f = np.zeros((hidden_size, 1))

        # 輸入門權重與偏置
        self.w_i = np.random.randn(hidden_size, hidden_size + input_size)
        self.b_i = np.zeros((hidden_size, 1))

        # 輸出門權重與偏置
        self.w_o = np.random.randn(hidden_size, hidden_size + input_size)
        self.b_o = np.zeros((hidden_size, 1))

        # 記憶細胞權重與偏置
        self.w_c = np.random.randn(hidden_size, hidden_size + input_size)
        self.b_c = np.zeros((hidden_size, 1))

        # 短期記憶（隱藏狀態）
        self.h = np.zeros((hidden_size, 1))

        # 長期記憶（記憶單元狀態）
        self.c = np.zeros((hidden_size, 1))

        # 門變量初始化
        self.f = None
        self.i = None
        self.c_tilde = None
        self.o = None
        self.combined = None

    def forward(self, x):

        # 結合上一時間步的隱藏狀態與當前輸入
        self.combined = np.vstack((self.h, x))

        self.c_prev = self.c.copy()

        # 遺忘門計算
        self.f = sigmoid(np.dot(self.w_f, self.combined) + self.b_f)

        # 輸入門計算
        self.i = sigmoid(np.dot(self.w_i, self.combined) + self.b_i)
        self.c_tilde = tanh(np.dot(self.w_c, self.combined) + self.b_c)

        # 記憶單元更新
        self.c = (self.f * self.c) + (self.i * self.c_tilde)

        # 輸出門計算
        self.o = sigmoid(np.dot(self.w_o, self.combined) + self.b_o)

        # 隱藏狀態更新
        self.h = self.o * tanh(self.c)

        return self.h
    
    def backward(self, d_h_next, d_c_next):
        # 輸出門梯度計算

        d_o = d_h_next * tanh(self.c)

        # 記憶單元梯度計算

        d_c = d_h_next * self.o * tanh_derivative(self.c)
        
        # 記憶單元候選值梯度計算

        d_c_tilde = d_h_next * self.o * tanh_derivative(self.c) * self.i

        # 輸入門梯度計算
        d_i = d_h_next * self.o * tanh_derivative(self.c)  
        d_i = d_i * self.c_tilde  
       
  

        # 遺忘門梯度計算
        
        d_f = d_h_next * self.o * tanh_derivative(self.c)  
        d_f = d_f * self.c_prev 


        # 計算每個門的權重和偏置的梯度 
        d_b_f = d_h_next * self.o * tanh_derivative(self.c) * self.c_prev * sigmoid_derivative(self.f)
        d_w_f = np.dot(d_b_f, self.combined.T)

        d_b_i = d_h_next * self.o * tanh_derivative(self.c) * self.c_tilde * sigmoid_derivative(self.i)
        d_w_i = np.dot(d_b_i, self.combined.T)

        d_b_c = d_h_next * self.o * tanh_derivative(self.c) * self.i * tanh_derivative(self.c_tilde)
        d_w_c = np.dot(d_b_c, self.combined.T)

        d_b_o = d_h_next * tanh(self.c) * sigmoid_derivative(self.o)
        d_w_o = np.dot(d_b_o, self.combined.T)

        # 更新每個門的權重和偏置
        self.w_f -= self.learning_rate * d_w_f
        self.b_f -= self.learning_rate * np.sum(d_b_f, axis=1, keepdims=True)
        self.w_i -= self.learning_rate * d_w_i
        self.b_i -= self.learning_rate * np.sum(d_b_i, axis=1, keepdims=True)
        self.w_c -= self.learning_rate * d_w_c
        self.b_c -= self.learning_rate * np.sum(d_b_c, axis=1, keepdims=True)
        self.w_o -= self.learning_rate * d_w_o
        self.b_o -= self.learning_rate * np.sum(d_b_o, axis=1, keepdims=True)
        
        # 將所有梯度結合，用於下一步反向傳播
        d_combined = np.dot(self.w_f.T, d_f) + np.dot(self.w_i.T, d_i) + \
                     np.dot(self.w_c.T, d_c_tilde) + np.dot(self.w_o.T, d_o)

        # 隱藏狀態梯度和記憶狀態梯度
        d_h_prev = d_combined[:self.hidden_size, :]  # 隱藏狀態梯度
        d_c_prev = d_h_next * self.o * tanh_derivative(self.c) * self.f  # 記憶狀態梯度

        # 回傳前一時間步的隱藏狀態和記憶狀態梯度
        return d_h_prev, d_c_prev
    
    def reset_state(self):
        self.h = np.zeros((self.hidden_size, 1))
        self.c = np.zeros((self.hidden_size, 1))
        

    
    def mse_loss(self, predictions, targets):
        return np.mean((predictions - targets) ** 2)

    def mse_loss_derivative(self, predictions, targets):
        return 2 * (predictions - targets)




# 測試案例

# 初始化 LSTM 參數
"""input_size = 2   # 假設每個輸入是 2 維向量
hidden_size = 4  # 隱藏層大小為 4
learning_rate = 0.01  # 學習率
lstm = LSTM(input_size, hidden_size, learning_rate)

# 生成隨機輸入數據 (一個 batch 的數據)
np.random.seed(42)  # 設置隨機種子
x = np.random.randn(input_size, 1)  # 單步輸入 2 維特徵 (2, 1) 的數據


# 定義目標輸出 (這裡假設隨機目標，實際中可以根據具體任務設定)
target = np.random.randn(hidden_size, 1)

# 訓練迴圈
epochs = 10000
for epoch in range(epochs):
    # 前向傳播
    output = lstm.forward(x)
    
    # 計算損失
    loss = lstm.mse_loss(output, target)
    
    # 計算損失的導數
    d_loss = lstm.mse_loss_derivative(output, target)
    
    # 反向傳播 (注意這裡初始的 d_h_next 和 d_c_next 為 0)
    d_h_next = d_loss
    d_c_next = np.zeros_like(lstm.c)
    
    lstm.backward(d_h_next, d_c_next)
    
    # 每 100 個 epoch 打印一次損失
    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}')

# 最後一個 epoch 的輸出和損失
print(f'Final Output:\n{output}')
print(f'Target Value:\n{target}')
print(f'Final Loss: {loss:.4f}')"""


# 测试案例
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 加載和預處理加州房價數據集
california_housing = fetch_california_housing()
X, y = california_housing.data[:1000], california_housing.target[:1000]

# 將數據集劃分為訓練集和測試集（80%訓練，20%測試）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 標準化輸入特徵和目標變量
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# 重塑輸入數據以適應 LSTM
# 將每個特徵視為序列中的一個時間步
sequence_length = X_train.shape[1]  # 8
input_size = 8  # 每个输入样本有8个特征
hidden_size = 50
learning_rate = 0.001
epochs = 500

# 初始化 LSTM
lstm = LSTM(input_size, hidden_size, learning_rate)

# 由於 LSTM 輸出的是 hidden_size，因此我們需要一個線性層將其映射到輸出
# 初始化線性層的權重和偏置
W_linear = np.random.randn(1, hidden_size) * np.sqrt(1 / hidden_size)
b_linear = np.zeros((1, 1))

def linear_forward(h):
    """
    h: (batch_size, hidden_size)
    Returns: (batch_size, 1)
    """
    return np.dot(h, W_linear.T) + b_linear.T  # (batch_size, 1)

def linear_backward(d_out, h):
    """
    d_out: (batch_size, 1)
    h: (batch_size, hidden_size)
    """
    global W_linear, b_linear
    d_W = np.dot(d_out.T, h)  # (1, hidden_size)
    d_b = np.sum(d_out, axis=0, keepdims=True).T  # (1, 1)
    d_h = np.dot(d_out, W_linear)  # (batch_size, hidden_size)
    # 更新線性層的參數
    W_linear -= learning_rate * d_W
    b_linear -= learning_rate * d_b
    return d_h

# 訓練循環
epoch_losses = []
for epoch in range(1, epochs + 1):
    epoch_loss = 0
    lstm.reset_state()  # 重置狀態，处理单个样本
    for i in range(len(X_train)):
        X_sample = X_train[i].reshape(-1, 1)  # 重塑輸入為 (input_size, 1)

        # 前向傳播
        h = lstm.forward(X_sample)  # (hidden_size, 1)

        # 前向傳播線性層
        y_pred = linear_forward(h.T)  # (1, 1)

        # 計算損失
        loss = np.mean((y_pred - y_train[i]) ** 2)
        epoch_loss += loss

        # 計算損失對 y_pred 的梯度
        d_loss = (2 * (y_pred - y_train[i]))  # (1,)

        # 反向傳播線性層
        d_loss = d_loss.reshape(1, 1)  # (1, 1)
        d_h = linear_backward(d_loss, h.T)  # (hidden_size,)

        # 反向傳播 LSTM
        d_c_last = 0  # 沒有來自未來時間步的梯度
        lstm.backward(d_h.reshape(-1, 1), d_c_last)

    # 計算每個 epoch 的平均損失
    avg_epoch_loss = epoch_loss / len(X_train)
    epoch_losses.append(avg_epoch_loss)

    # 每10個epoch輸出一次損失值
    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{epochs}, Training Loss: {avg_epoch_loss:.4f}")

average_mse_all_epochs = np.mean(epoch_losses)
print(f"Train MSE: {average_mse_all_epochs:.4f}")

    


# 評估測試集
# 直接處理所有測試數據
lstm.reset_state()
y_pred_test = []

for i in range(len(X_test)):
    X_sample = X_test[i].reshape(-1, 1)  # 重塑輸入為 (input_size, 1)
    h_test = lstm.forward(X_sample)  # (hidden_size, 1)
    
    # 前向傳播線性層
    y_pred = linear_forward(h_test.T)  # (1, 1)
    y_pred_test.append(y_pred.flatten()[0])  # 提取标量值并添加到列表中

# 将预测结果转为数组
y_pred_test = np.array(y_pred_test)  # (batch_size_test,)

# 反標準化預測值和真實值
test_predictions = scaler_y.inverse_transform(y_pred_test.reshape(-1, 1)).flatten()
y_test_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

# 計算測試集上的均方誤差
mse = np.mean((test_predictions - y_test_true) ** 2)
print(f"Test MSE: {mse:.4f}")

for i in range(10):
    print(f"樣本 {i + 1}: 預測值 = {test_predictions[i]:.2f}, 實際值 = {y_test_true[i]:.2f}")


plt.figure(figsize=(10, 5))
plt.plot(epoch_losses, label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.title("Training Loss Over Epochs")
plt.legend()
plt.show()


plt.figure(figsize=(10, 5))
plt.plot(y_test_true[:50], label="Actual Values", marker='o', linestyle="--")
plt.plot(test_predictions[:50], label="Predicted Values", marker='x', linestyle="-")
plt.xlabel("Sample")
plt.ylabel("House Price")
plt.title("Comparison of Actual and Predicted Values")
plt.legend()
plt.show()