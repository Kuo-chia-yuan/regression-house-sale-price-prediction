# regression-house-sale-price-prediction

## 介紹
手刻 Fully Connection Layer，預測房價

## 作法說明與寫法
依照 code 順序進行講解：
1. 讀取 train-v3.csv、valid-v3.csv、test-v3.csv 轉成 train_data、valid_data、test_data

3. 將 train_data 中的 'id'、'price' 和 13 項與房價無關的參數移除
  
5. 利用 StandardScaler() 和 .fit_transform() 將 data 標準化，使不同特徵的值縮放到相同的範圍，讓之後的訓練能夠收斂更快，同時避免梯度爆炸或梯度消失
  
7. 建立 4 層 Fully Connection Layer，激勵函數採用 ReLU
   
9. Loss function 採用 L1 Loss (即 MAE)，優化器採用 Adam，學習率為 0.003
    
11. training data 迭代共 500 次
    
13. 訓練完成後，將valid_data 放進 model 預測 price，根據 MAE Loss 進一步調整參數
    
15. 將 test_data 放進 model 預測 price，並將數據結果轉成 numpy，再存進 submission.csv

## Network 架構
```
class Net(nn.Module): 
  def __init__(self, input_size):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(input_size, 128)
    self.fc2 = nn.Linear(128, 64)
    self.fc3 = nn.Linear(64, 32)
    self.fc4 = nn.Linear(32, 1)

  def forward(self, x):
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = torch.relu(self.fc3(x))
    x = self.fc4(x)
    return x
```

## 程式方塊圖

  ![image](https://github.com/Machine-Learning-NYCU/regression-house-sale-price-prediction-challenge-312605018CYKuo/assets/144798019/7768731c-397d-4bac-affa-c52796d282c6)


## loss function 收斂曲線圖

  ![image](https://github.com/Machine-Learning-NYCU/regression-house-sale-price-prediction-challenge-312605018CYKuo/assets/144798019/52f395f0-535d-42e2-af84-44a363d7a518)


## 畫圖做結果分析
和 price 有明顯相關性的房屋參數共有以下6項：
bedrooms、bathrooms、sqft_living、grade、sqft_above、sqft_living15

其餘房屋參數和 price 無直接相關性
- bedrooms

  ![image](https://github.com/MachineLearningNYCU/regression-house-sale-price-prediction-challenge-312605018CYKuo/assets/144798019/d35161bc-435d-46b4-9f83-c61196ad9105)
- bathrooms

  ![image](https://github.com/MachineLearningNYCU/regression-house-sale-price-prediction-challenge-312605018CYKuo/assets/144798019/fa16fb52-22ee-423a-ab16-1050566eafcd)
  
- sqft_living

  ![image](https://github.com/MachineLearningNYCU/regression-house-sale-price-prediction-challenge-312605018CYKuo/assets/144798019/fd0652e7-3de0-42b1-895f-48ad138a07ca)

- grade

  ![image](https://github.com/MachineLearningNYCU/regression-house-sale-price-prediction-challenge-312605018CYKuo/assets/144798019/1e91b7d4-a235-4d90-9923-a0f6ff3ff558)

- sqft_above

  ![image](https://github.com/MachineLearningNYCU/regression-house-sale-price-prediction-challenge-312605018CYKuo/assets/144798019/9b183642-a6ef-4295-b90a-0fc4b9be5167)

- sqft_living15

  ![image](https://github.com/MachineLearningNYCU/regression-house-sale-price-prediction-challenge-312605018CYKuo/assets/144798019/79afe1d3-51a3-4e0c-81c1-c60a5588805c)

## 討論預測值誤差很大的，是怎麼回事？如何改進？
train_data 的 loss 約為 124970  
valid_data 的 loss 約為 210200
- 兩者有明顯的差距，我認為這是 overfitting 的現象，因此我做出以下的改進方式：
  1. Fully Connected 從 6 層改成 4 層，避免參數過度訓練
  2. 訓練時的迭代次數從 1000 次降低成 500 次
  3. 將 Learning Rate 降低至 0.003
  4. 多嘗試其他激勵函數，設法找出效果最好、最適合此 model 的激勵函數
