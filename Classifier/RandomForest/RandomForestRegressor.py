import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv("dataset/insurance.csv")
df = pd.get_dummies(df, columns=["sex", "smoker", "region"], drop_first=True)

y = df['charges']
x = df.drop(columns=["charges"])

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=22, train_size=0.7)

lr = LinearRegression()
model_lr = lr.fit(x_train, y_train)
print("score", model_lr.score(x_test, y_test))
print("")

rf = RandomForestRegressor(n_estimators=200)
model_rf = rf.fit(x_train, y_train)
print("score", model_rf.score(x_test, y_test))
print("")
print("")
print("")

"""
age - bmi - children - sex_male - smoker_yes - region_northwest - region_southeast - region_southwest

"""
age = 21
bmi = 35
children = 0
sex_male = 1
smoker_yes = 0
region_northwest = 0
region_southeast = 1
region_southwest = 0

people = [[age, bmi, children, sex_male, smoker_yes, region_northwest, region_southeast, region_southwest]]
print("Tahminler")
print("")
print("model_lr tahmin:", model_lr.predict(people))
print("")
print("model_rf tahmin:", model_rf.predict(people))
print("")
print("")