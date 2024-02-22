# Gerekli kütüphaneleri içe aktar
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Veri setini yükle
df = pd.read_csv("dataset/heart.csv")

# Hedef değişken ve özellikler matrisini tanımla
y = df["output"]
x = df.drop("output", axis=1)

# Veriyi eğitim ve test alt kümelerine böle
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, train_size=0.6)

# Decision Tree modelini oluştur ve eğit
dt = DecisionTreeClassifier()
model_dt = dt.fit(x, y)
# Modelin doğruluk puanını yazdır
print("Decision Tree model score:", dt.score(x, y))
print("")

# Eğitim kümesi üzerinde Decision Tree modelini oluştur ve eğit
dt_train = DecisionTreeClassifier()
model_dt_train = dt_train.fit(x_train, y_train)
# Modelin eğitim kümesi üzerindeki doğruluk puanını yazdır
print("Decision Tree model train score:", dt_train.score(x_train, y_train))
print("")

# Random Forest modelini oluştur, eğitim kümesi üzerinde eğit ve test kümesi üzerinde değerlendir
rf = RandomForestClassifier(n_estimators=200)
model_rf = rf.fit(x_train, y_train)
# Modelin test kümesi üzerindeki doğruluk puanını yazdır
print("Random Forest model score:", model_rf.score(x_test, y_test))
print("")

# XGBoost modelini oluştur, eğitim kümesi üzerinde eğit ve test kümesi üzerinde değerlendir
xgb_model = xgb.XGBClassifier()
model_xgb = xgb_model.fit(x_train, y_train)
# Modelin test kümesi üzerindeki doğruluk puanını yazdır
print("XGBoost model score:", model_xgb.score(x_test, y_test))
print("")

# Rastgele bir insanın verilerini oluştur
age = 21
sex = 1
cp = 0
trtbps = 120
chol = 126
fbs = 0
restecg = 0
thalachh = 71
exng = 0
oldpeak = 1
slp = 1
caa = 3
thall = 1
people_inp = [[age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall]]

# Veri setinden rastgele bir insanın verilerini getir (output sütunu çıkar)
people = df.sample().drop("output", axis=1).values
print("Rastgele insan verisi:", people)
print("")

# Her bir modelin rastgele insan verisini tahmin etmesi
print("Decision Tree model tahmini:", model_dt.predict(people_inp))
print("Decision Tree model train tahmini:", model_dt_train.predict(people_inp))
print("Random Forest model tahmini:", model_rf.predict(people_inp))
print("XGBoost model tahmini:", model_xgb.predict(people_inp))
print("")

