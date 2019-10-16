import statsmodels.tsa.arima_model as arima
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

tmp = []
with open('TS.txt','r') as txt_file:
    N = int(txt_file.readline())
    for i in range(N):
        tmp.append(float(txt_file.readline()))


steps=30
data=pd.DataFrame(data=tmp,columns=['all_data'])
train_data = pd.DataFrame(data=tmp[0:len(tmp)-steps],columns=['ts'])
test_data= pd.DataFrame(data=tmp[len(tmp)-steps:],columns=['ts'])
train_data_diff=train_data.diff().dropna()
model = arima.ARIMA(train_data_diff,order=(5, 1, 2)).fit(disp=0)

forcasts=model.forecast(steps)[0]

for i in range(steps):
    forcasts[i] = forcasts[i] + data.values[-steps+i]


pred= pd.DataFrame(data=forcasts,columns=['forcasts'])
from sklearn.metrics import mean_squared_error
err =mean_squared_error(test_data.values,pred.values)
pred_test=pd.concat([pred,test_data],axis=0,sort=False)
pred_test.plot()
plt.show()
a=1