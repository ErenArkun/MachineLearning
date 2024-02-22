#Overfitting -> Aşırı öğrenme = Ezbere
#Underfitting -> Az öğrenme
#Balanced Fitting -> Dengele öğrenme

import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df=sns.load_dataset("diamonds")

df = pd.get_dummies(df, columns=['cut', 'color', 'clarity'], drop_first=True)
#df = pd.get_dummies(df, columns=['cut', 'color', 'clarity'])

print(df.head(3))
print("")

y = df[['price']]
x = df.drop("price", axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=19, train_size=0.76)

lm = LinearRegression()
model = lm.fit(x_train, y_train)

print("test")
print(model.score(x_test, y_test))
print("")

print("train")
print(model.score(x_train, y_train))
print("")