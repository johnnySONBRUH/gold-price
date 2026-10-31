from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score, mean_squared_error, root_mean_squared_error
import pandas as pd
data = pd.read_csv("data.csv",index_col=False)
data['Date'] = pd.to_datetime(data['Date'])
data['Date'] = (data['Date'] - pd.Timestamp('1970-01-01')) // pd.Timedelta('1s')
X = data
X = X.drop(columns=['USD','EUR','JPY','GBP','CAD','CHF','INR',
'CNY','TRY','SAR','IDR','AED','THB','VND','EGP','KRW','RUB','ZAR','AUD'])
Y = data['USD'].apply(lambda x: x.replace(',',''))
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
print(X,Y)
model = RandomForestRegressor(random_state=42,n_estimators=500)
model.fit(x_train,y_train)
pred = model.predict(x_test)
mae = mean_absolute_error(y_test,pred)
mape = mean_absolute_percentage_error(y_test,pred)
r2 = r2_score(y_test,pred)
mse = mean_squared_error(y_test,pred)
rmse = root_mean_squared_error(y_test,pred)
print(f'MAE: {mae}')
print(f'MAPE: {mape}')
print(f'R2: {r2}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')