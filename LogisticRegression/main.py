import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("dataset/card.csv")
df=df.drop("ID", axis=1)

y = df["default.payment.next.month"]
x = df.drop("default.payment.next.month", axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.77, random_state=6)

log = LogisticRegression()
#log = LogisticRegression(max_iter=1000, solver='sag')
model = log.fit(x_train, y_train)

skr = model.score(x_test, y_test)

print("skor:",skr)
print("")

denemex = np.array(x.iloc[2015])
print("sonuc X:", model.predict([denemex]))
print("")

#denemey = np.array(y.iloc[1903])
#print("sonuc Y:", model.predict(denemey))
print("sonuc Y:",  y.iloc[2015])
print("")
