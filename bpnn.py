import numpy as np
class BPNN(object):
    def __init__(self , dataset , learning_rate = 0.001 , n_iter = 10000 , momentum = 0.9 , shutdown_condition = 0.01):
        self.n_iter = n_iter        #迭代次數:控制訓練的次數，高迭代 -> 高精度 但可能會過度擬合
        self.dataset = dataset      #資料集
        self.learning_rate = learning_rate  #學習律:決定每次更新權重的步伐大小 過大過小都會影響模型收斂
        self.x = dataset.train_x    #訓練特徵
        self.Y = dataset.train_Y    #訓練標籤
        self.shutdown_condition = shutdown_condition    #停止條件:設定模型停止訓練的閾值 若成本函數的變化量 < 該閾值 則停止訓練
        self.cost = []              #儲存訓練過程中的成本值:衡量模型預測與實際結果的差距 backpropagation的目的即是最小化成本值
        self.momentum = momentum    #動量值:加速梯度下降收斂速度，減少震盪，提升訓練效率
        self.setup()

    def setup(self):
        self.set_nn_architecture()
        self.set_weight()
    
    # step1
    def set_nn_architecture(self):
        self.input_node = self.x.shape[1]   #設置輸入層的節點數量 shape[1]:讀取矩陣的行數(直的)
        self.output_node = self.Y.shape[1]  #設置輸出層的節點數量 shape[1]:讀取矩陣的行數(直的)
        self.hidden_node = int((self.input_node + self.output_node) / 2)    #設置隱藏層的節點數量 將輸入層與輸出層的節點數量相加除2

        # bias : 設置偏置 1.提高模型表達力 2.增加模型靈活性 3.保證sigmoid函數在非線性區域作用 4.防止模型過擬合
        self.h_b = np.random.random(self.hidden_node) * 0.3 + 0.1   #設置隱藏層的bias出來會是一個 self.hidden_node 長度的陣列
        self.y_b = np.random.random(self.output_node) * 0.3 + 0.1   #設置輸出層的bias出來會是一個 self.output_node 長度的陣列
        #乘0.3是為了將結果控制在0 ~ 0.3之間 再加上0.1是為了將結果控制在 0.1 ~ 0.4之間 (不一定要設這樣 看自己的經驗來決定)
    
    # step2
    def set_weight(self):
        self.w1 = np.random.random((self.input_node , self.hidden_node))    #隨機初始化輸入層到隱藏層的權重
        self.w2 = np.random.random((self.hidden_node , self.output_node))   #隨機初始化隱藏層到輸出層的權重
    
    # step3
    def predict(self , x , Y):  #正向傳播 x:輸入的特徵資料 y:真實標籤資料
        self.h = self.sigmoid((np.dot(x , self.w1) + self.h_b))     #計算隱藏層的輸出 = sigmoid( (輸入的資料 與 權重w1 的點積) + 加隱藏層的偏置 h_b )        
        self.y = self.sigmoid((np.dot(self.h , self.w2) + self.y_b))    #計算輸出層的輸出 = sigmoid( (隱藏層輸出h 與 權重w2 的點積) + 加輸出層的偏置 y_b )
        zy = np.where(self.y > 0.5 , 1 , 0)     #將輸出層的輸出值 轉為二進制分類結果 若該值 > 0.5 則設為 1 其餘設為0
        p_y = Y - zy        #計算真實結果與預測結果之間的誤差p_y
        self.acc = 0        #準確度計數器
        for i in p_y:       #遍歷p_y的每一個
            if (i.sum() == 0):  #若每個合為0 即 全部皆為0 因前面已經做過二進制分類
                self.acc += 1   #準確度計數器+1
        self.acc = self.acc / Y.shape[0] * 100.0 #計算準確度 shape[0]:讀取矩陣的列數(橫的) 乘100是為了轉成百分比
        return self
    
    # step4
    def backend(self):  #反向傳播
        E = (self.Y - self.y)   #計算真實值(Y)與預測值(y)的誤差
        errors = np.sum(np.square(E)) / self.Y.shape[1] / self.Y.shape[0]   
        #計算均方誤差 np.square(E):將誤差平方, np.sum:將誤差平方相加,  self.Y.shape[1]:樣本特徵數量, self.Y.shape[0] :樣本數量
        
        #### 輸出層 delta 計算
        delta_y = E * self.y * (1 - self.y) 
        #計算輸出層誤差 :E * sigmoid的倒數(y * (1 - y)) , self.y 帶入 y
        
        ### 隱藏層 delta 計算
        delta_h = (1 - self.h) * self.h * np.dot(delta_y , self.w2.T)
        #計算隱藏層的誤差 : 輸出層誤差 與 隱藏層到輸出層權重w2的轉置 的點積 * sigmoid倒數(y * (1 - y)) , self.h(隱藏層的輸出) 帶入 y
        
        # self.w2 += self.learning_rate * self.h.T.dot(delta_y) + self.momentum * self.h.T.dot(delta_y)
        # self.w1 += self.learning_rate * self.x.T.dot(delta_h) + self.momentum * self.x.T.dot(delta_h)
        
        self.w2 += self.learning_rate * self.h.T.dot(delta_y)   #更新隱藏層到輸出層的權重: 學習律 * 隱藏層輸出的轉置 與 輸出層誤差 的點積
        self.w1 += self.learning_rate * self.x.T.dot(delta_h)   #更新輸入層到隱藏層的權重: 學習律 * 輸入的資料的轉置 與 隱藏層誤差 的點積
        self.y_b = self.learning_rate * delta_y.sum()       #更新輸出層偏置(bias): 學習律 * 輸出層誤差的總和
        self.h_b = self.learning_rate * delta_h.sum()       #更新隱藏層偏置(bias): 學習律 * 隱藏層誤差的總和
        return errors

    def train(self):
        self.error = 0      #儲存每次迭代的誤差
        for _iter in range(0 , self.n_iter):        #進行self.n_iter次迭代
            self.predict(self.x , self.Y)           #調用向前傳播
            self.error = self.backend()             #調用反向傳播 並將回傳的誤差errors 儲存
            self.cost.append(self.error)            #將誤差值存入self.cost []
            # if (_iter % 1000 == 0):
            #     print("Accuracy：%.2f" % self.acc)
            # 打印每次迭代的損失函數值
            print(f"Iteration {_iter+1}/{self.n_iter}, Error: {self.error}, Accuracy: {self.acc}%")
            if (self.acc >= 98):                    #若準確度計數器 >= 98% 則停止訓練
                return self
        return self

    def test(self):
        self.predict(self.dataset.test_x , self.dataset.test_Y)
        return self

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))