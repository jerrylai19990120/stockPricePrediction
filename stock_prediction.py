from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd

df = pd.read_csv('AADR.csv')
df.set_index(pd.DatetimeIndex(df['Date'].values), inplace=True)
df.index.name = "Date"
df.drop(columns='Date', inplace=True)
df['Price_up'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

x = df.iloc[:, 0:df.shape[1]-1].values
y = df.iloc[:, df.shape[1]-1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = DecisionTreeClassifier()
model.fit(x_train, y_train)

report= model.score(x_test, y_test)

df2 = pd.read_csv('AAAU.csv')
df2.set_index(pd.DatetimeIndex(df2["Date"].values), inplace=True)
df2.drop(columns=['Date'], inplace=True)
df2["Price_up"] = np.where(df2['Close'].shift(-1) > df2['Close'], 1, 0)
input_data = df2.iloc[:, 0:df2.shape[1]-1].values
answers = df2.iloc[:, df2.shape[1]-1].values
predictions = model.predict(input_data)

score = model.score(input_data, answers)
print(f"Model score is {score*100}%.")