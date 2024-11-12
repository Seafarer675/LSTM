from matplotlib import pyplot as plt
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

class LSTM:
    def __init__(self, input_size, hidden_size, learning_rate, batch_size):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # 初始化各門的權重和偏置
        self.w_f = np.random.randn(hidden_size, hidden_size + input_size)
        self.b_f = np.zeros((hidden_size, 1))
        self.w_i = np.random.randn(hidden_size, hidden_size + input_size)
        self.b_i = np.zeros((hidden_size, 1))
        self.w_o = np.random.randn(hidden_size, hidden_size + input_size)
        self.b_o = np.zeros((hidden_size, 1))
        self.w_c = np.random.randn(hidden_size, hidden_size + input_size)
        self.b_c = np.zeros((hidden_size, 1))

        # 初始化隱藏狀態和記憶單元狀態，支援批次大小
        self.h = np.zeros((hidden_size, batch_size))
        self.c = np.zeros((hidden_size, batch_size))

    def forward(self, x):
        # 確保 x 的形狀為 (input_size, batch_size)
        batch_size = x.shape[1]  # 獲取批次大小
        if self.h.shape[1] != batch_size:
            self.h = np.zeros((self.hidden_size, batch_size))  # 初始化批次大小的隱藏狀態
            self.c = np.zeros((self.hidden_size, batch_size))  # 初始化批次大小的記憶狀態

        # 結合上一時間步的隱藏狀態與當前輸入
        self.combined = np.vstack((self.h, x))  # (hidden_size + input_size, batch_size)

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
        # 輸出門梯度
        d_o = d_h_next * tanh(self.c)
        
        # 記憶單元梯度
        d_c = d_h_next * self.o * tanh_derivative(self.c) + d_c_next

        # 各門的梯度
        d_f = d_c * self.c * sigmoid_derivative(self.f)  # 遺忘門
        d_i = d_c * self.c_tilde * sigmoid_derivative(self.i)  # 輸入門
        d_c_tilde = d_c * self.i * tanh_derivative(self.c_tilde)  # 候選記憶
        d_b_f, d_b_i, d_b_c, d_b_o = map(lambda g: np.sum(g, axis=1, keepdims=True), (d_f, d_i, d_c_tilde, d_o))

        # 各權重梯度
        d_w_f = np.dot(d_f, self.combined.T)
        d_w_i = np.dot(d_i, self.combined.T)
        d_w_c = np.dot(d_c_tilde, self.combined.T)
        d_w_o = np.dot(d_o, self.combined.T)

        # 更新權重和偏置
        self.w_f -= self.learning_rate * d_w_f
        self.b_f -= self.learning_rate * d_b_f
        self.w_i -= self.learning_rate * d_w_i
        self.b_i -= self.learning_rate * d_b_i
        self.w_c -= self.learning_rate * d_w_c
        self.b_c -= self.learning_rate * d_b_c
        self.w_o -= self.learning_rate * d_w_o
        self.b_o -= self.learning_rate * d_b_o

        # 將前一層的梯度返回
        d_combined = np.dot(self.w_f.T, d_f) + np.dot(self.w_i.T, d_i) + \
                     np.dot(self.w_c.T, d_c_tilde) + np.dot(self.w_o.T, d_o)
        d_h_prev = d_combined[:self.hidden_size, :]
        d_c_prev = d_c * self.f

        return d_h_prev, d_c_prev

    def reset_state(self):
        # 每個批次重置狀態
        self.h = np.zeros((self.hidden_size, self.batch_size))
        self.c = np.zeros((self.hidden_size, self.batch_size))

    def mse_loss(self, predictions, targets):
        # 平均均方誤差損失
        return np.mean((predictions - targets) ** 2, axis=1)
    
    def mse_loss_derivative(self, predictions, targets):
        # 平均均方誤差的導數
        return 2 * (predictions - targets)


#測試
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 加載和預處理加州房價數據集
california_housing = fetch_california_housing()
X, y = california_housing.data[:10000], california_housing.target[:10000]

# 將數據集劃分為訓練集和測試集（80%訓練，20%測試）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 標準化輸入特徵和目標變量
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# 設置 LSTM 的參數
sequence_length = X_train.shape[1]  # 8
input_size = 8  # 每個輸入樣本有8個特徵
hidden_size = 50
learning_rate = 0.001
epochs = 300
batch_size = 32  # 設置批次大小

# 初始化 LSTM
lstm = LSTM(input_size, hidden_size, learning_rate, batch_size)

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
num_batches = len(X_train) // batch_size  # 計算批次數

for epoch in range(1, epochs + 1):
    epoch_loss = 0
    
    # 隨機打亂訓練數據
    permutation = np.random.permutation(len(X_train))
    X_train_shuffled = X_train[permutation]
    y_train_shuffled = y_train[permutation]

    for b in range(num_batches):
        # 每批次的數據
        start = b * batch_size
        end = start + batch_size
        X_batch = X_train_shuffled[start:end].T  # (input_size, batch_size)
        y_batch = y_train_shuffled[start:end]  # (batch_size,)

        # 重置 LSTM 狀態
        lstm.reset_state()
        
        # LSTM 前向傳播
        h = lstm.forward(X_batch)  # (hidden_size, batch_size)
        
        # 線性層前向傳播
        y_pred = linear_forward(h.T).flatten()  # (batch_size,)

        # 計算損失並累加
        loss = np.mean((y_pred - y_batch) ** 2)
        epoch_loss += loss

        # 損失對 y_pred 的梯度
        d_loss = (2 * (y_pred - y_batch) / batch_size).reshape(-1, 1)  # (batch_size, 1)

        # 線性層反向傳播
        d_h = linear_backward(d_loss, h.T)

        # LSTM 反向傳播
        d_c_last = np.zeros((hidden_size, batch_size))
        lstm.backward(d_h.T, d_c_last)

    # 計算每個 epoch 的平均損失
    avg_epoch_loss = epoch_loss / num_batches
    epoch_losses.append(avg_epoch_loss)

    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{epochs}, Training Loss: {avg_epoch_loss:.4f}")

average_mse_all_epochs = np.mean(epoch_losses)
print(f"Train MSE: {average_mse_all_epochs:.4f}")

# 評估測試集
lstm.reset_state()
y_pred_test = []

for i in range(0, len(X_test), batch_size):
    X_batch_test = X_test[i:i + batch_size].T  # (input_size, batch_size)
    h_test = lstm.forward(X_batch_test)
    y_pred_batch = linear_forward(h_test.T).flatten()  # 批次內的預測
    y_pred_test.extend(y_pred_batch)

# 反標準化預測值和真實值
test_predictions = scaler_y.inverse_transform(np.array(y_pred_test).reshape(-1, 1)).flatten()
y_test_true = scaler_y.inverse_transform(y_test[:len(test_predictions)].reshape(-1, 1)).flatten()

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



