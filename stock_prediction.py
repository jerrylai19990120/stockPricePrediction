from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd

df = pd.read_csv('stockPrice.csv')
df.set_index(pd.DatetimeIndex(df['Date'].values), inplace=True)
df.index.name = "Date"
df.drop(columns='Date', inplace=True)
df['Price_up'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

x = df.iloc[:, 0:7].values
y = df.iloc[:, 7].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = DecisionTreeClassifier()
model.fit(x_train, y_train)

report= model.score(x_test, y_test)

result = model.predict(x_test)
print(report)
print(result)
