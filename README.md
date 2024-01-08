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

  ![image](https://github.com/Kuo-chia-yuan/regression-house-sale-price-prediction/assets/56677419/1943d339-e0cd-485f-9bdf-52d2226caa39)

## loss function 收斂曲線圖

  ![image](https://github.com/Kuo-chia-yuan/regression-house-sale-price-prediction/assets/56677419/17f590fc-e5a8-4bbd-897e-7f1bdca86d6e)

## 畫圖做結果分析
和 price 有明顯相關性的房屋參數共有以下6項：
bedrooms、bathrooms、sqft_living、grade、sqft_above、sqft_living15

其餘房屋參數和 price 無直接相關性
- bedrooms

  ![image](https://github.com/Kuo-chia-yuan/regression-house-sale-price-prediction/assets/56677419/7e417b99-05bc-44d9-a4a0-384bfab4eee2)

- bathrooms

  ![image](https://github.com/Kuo-chia-yuan/regression-house-sale-price-prediction/assets/56677419/5f3baee7-3127-4133-99fd-c7ce8869cec2)

- sqft_living

  ![image](https://github.com/Kuo-chia-yuan/regression-house-sale-price-prediction/assets/56677419/a5c8281e-1af1-4f9a-ad0d-55863b96309a)

- grade

  ![image](https://github.com/Kuo-chia-yuan/regression-house-sale-price-prediction/assets/56677419/8e52448c-2be1-41ff-8358-dafe0eead0f4)

- sqft_above

  ![image](https://github.com/Kuo-chia-yuan/regression-house-sale-price-prediction/assets/56677419/ffb00b87-03cb-4245-b664-bc4c6784ece2)

- sqft_living15

  ![image](https://github.com/Kuo-chia-yuan/regression-house-sale-price-prediction/assets/56677419/1af84bd3-a811-4d34-b994-243eda6cdfb0)

## 討論預測值誤差很大的，是怎麼回事？如何改進？
train_data 的 loss 約為 124970  
valid_data 的 loss 約為 210200
- 兩者有明顯的差距，我認為這是 overfitting 的現象，因此我做出以下的改進方式：
  1. Fully Connected 從 6 層改成 4 層，避免參數過度訓練
  2. 訓練時的迭代次數從 1000 次降低成 500 次
  3. 將 Learning Rate 降低至 0.003
  4. 多嘗試其他激勵函數，設法找出效果最好、最適合此 model 的激勵函數
