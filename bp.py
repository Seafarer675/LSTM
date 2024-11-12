import numpy as np
from sklearn.datasets import load_iris
"""data = {"你": [1, 0, 0], "好": [0, 1, 0], "嗎": [0, 0, 1]}
x = np.array((data["嗎"], data["你"], data["好"]), dtype=float)
y = np.array((data["你"], data["好"], data["嗎"]), dtype=float)"""
'''x = np.array(([2, 9], [1, 5], [3, 6]), dtype=float) #輸入的樣本 每行(橫的)代表一個輸入樣本 此範例共三個 3X2
y = np.array(([92], [86], [89]), dtype=float)       #實際的輸出的樣本 每行(橫的)代表一個輸出樣本 此範例共三個 3X1'''

class NeuralNetwork(object):

    def __init__(self):     #設置神經網路的初始參數
        self.inputSize = 4  #設置輸入層大小
        self.outputSize = 3 #設置輸出層大小
        self.hiddenSize = 4  #設置隱藏層大小

        self.w1 = np.random.randn(self.inputSize, self.hiddenSize)  #隨機初始化權重w1 輸入層到隱藏層 2X3
        self.w2 = np.random.randn(self.hiddenSize, self.outputSize) #隨機初始化權重w2 隱藏層到輸出層 3X1

    def sigmoid(self, s, deriv = False):    #sigmoid激活函數 根據deriv返回導數或值 s:包含輸入值的numpy_array
        if(deriv == True):
            return s * (1 - s)              #回傳sigmoid函數的導數
        return 1 / (1 + np.exp(-s))         #回傳sigmoid函數值 np.exp()返回e常數的冪次方 e:2.1828
    
    
    def feedForward(self, x):               #x:輸入數據 x:是一個包含多個樣本的numpy_array
        
        self.z = np.dot(x, self.w1)         #1.輸入層 ~ 隱藏層:x * 輸入層到隱藏層的權重:self.w1 = self.z : 隱藏層的輸入

        self.z2 = self.sigmoid(self.z)      #將隱藏層的輸入以sigmoid函數激活後變為隱藏層的輸出:self.z2
        
        self.z3 = np.dot(self.z2, self.w2)  #2.隱藏層 ~ 輸出層:self.z2 * 隱藏層到輸出層的權重:self.w2 = self.z3
        
        output = self.sigmoid(self.z3)      #將self.z3以sigmoid函數激活後得到最終結果
        
        return output
    
    def backward(self, x, y, output):       #用於計算梯度並更新權重
       
        self.output_error = y - output      #計算輸出層的誤差 = 真實值 - 預測值
        self.output_delta = self.output_error * self.sigmoid(output, deriv=True) #將輸出層的誤差轉為梯度: 誤差 * sigmoid函數的導數 = 梯度

        self.z2_error = self.output_delta.dot(self.w2.T)    #計算隱藏層的誤差:輸出層的誤差梯度(self.output_delta) * 隱藏層到輸出層權種的轉置矩陣
        self.z2_delta = self.z2_error * self.sigmoid(self.z2, deriv=True)   #將隱藏層的誤差轉為梯度: 誤差 * sigmoid函數的導數 = 梯度

        print(self.output_delta)
        print(self.z2.T)
        
        self.w1 += x.T.dot(self.z2_delta)               #更新輸入層到隱藏層的權重: 原始權重 + (輸入數據的轉置矩陣 * 隱藏層的誤差梯度)
        self.w2 += self.z2.T.dot(self.output_delta)     #更新隱藏層到輸出層的權重: 原始權重 + (隱藏層的輸出的轉置矩陣 * 輸出層的誤差梯度)
        



    def train(self, x, y):
        
        output = self.feedForward(x)        #計算出模型對每個輸入的「預測輸出」
        
        self.backward(x, y, output)         #根據真實值與預測值的差異計算出梯度並更新權重

    def printLoss(self, y, x):
        print("Loss: " + str(np.mean(np.square(y - nn.feedForward(x)))))    #np.square():將array內的每個元素都取平方 ; np.mean(): 求該array內所有元素的均值

    def printPredicted_Output(self, z):
        print("Predicted Output: " + str(nn.feedForward(z)))

nn = NeuralNetwork()
'''x = np.array(([2, 9], [1, 5], [3, 6]), dtype=float) #輸入的樣本 每行(橫的)代表一個輸入樣本 此範例共三個 3X2
y = np.array(([92], [86], [89]), dtype=float)
x = x/np.amax(x, axis=0)    #進行正規化 將值縮放到 0 ~ 1 之間
y = y/100                   #進行正規化 將值縮放到 0 ~ 1 之間

for i in range(20000):
    if(i % 10000 == 0):
        nn.printLoss(y, x)
    nn.train(x, y)
print("Input: " + str(x))
print("Actual Output: " + str(y))

print("\n")

nn.printPredicted_Output(x)'''
#print("Predicted Output: " + str(nn.feedForward(x)))


#Q1:隱藏層的輸入以sigmoid函數激活後變為隱藏層的輸出? A: 隱藏層的輸入為各輸入層輸出的加總 其值可能會超過1 故用sigmoid函數控制其在 0 ~ 1 範圍內
#Q2:為何要將self.z3以sigmoid函數激活後才能得到最終結果? A: 輸出層的輸入為各隱藏層輸出的加總 其值可能會超過1 故用sigmoid函數控制其在 0 ~ 1 範圍內
#Q3:誤差 * sigmoid函數的導數 = 梯度 ? A: 沒錯 公式
#Q4:Loss值怎麼算 A: 均方誤差  loss值越小越好 -> 模型的預測值會接近真實值
