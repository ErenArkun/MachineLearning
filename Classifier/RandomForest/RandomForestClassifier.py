# Gerekli kütüphaneleri içe aktarın
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# CSV dosyasını okuyun
df = pd.read_csv("dataset/heart.csv")

# Bağımsız ve bağımlı değişkenleri ayırın
y = df["output"]
x = df.drop("output", axis=1)

# Veriyi eğitim ve test setlerine ayırın
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=16, train_size=0.7)

# Random Forest sınıflandırıcı modelini oluşturun
forest = RandomForestClassifier(n_estimators=200, max_depth=4)

# Modeli tüm veri üzerinde eğitin ve doğruluk skorunu yazdırın
model = forest.fit(x, y)
frstSkr = model.score(x, y)
print(frstSkr)
print("")

"""
  age  - sex  -  cp  -  trtbps -  chol    - fbs  -  restecg - thalachh  -  exng - oldpeak  - slp  - caa  - thall
29, 77 - 0, 1 - 0, 3 - 94, 200 - 126, 564 - 0, 1 -   0, 2   -  71, 202  -  0, 1 -  0, 6.2  - 0, 2 - 0, 4 -  0, 3
"""

age = 22
sex = 1
cp = 2
trtbps = 136
chol = 130
fbs = 0
restecg = 2
thalachh = 80
exng = 0
oldpeak = 5
slp = 1
caa = 3
thall = 1

person = [[age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall]]

print("overfit tahmin", model.predict(person))
print("")

# Modeli eğitim seti üzerinde eğitin ve test seti üzerinde doğruluk skorunu yazdırın
modelTrain = forest.fit(x_train, y_train)
skorTest = model.score(x_test, y_test)
print(skorTest)
print("")

print("Train tahmin", modelTrain.predict(person))
print("")