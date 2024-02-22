import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# CSV dosyasından veriyi oku
df = pd.read_csv("dataset/plane.csv")

# DataFrame hakkında bilgiyi yazdır
print(df.info())
print("")

# İlgilenilen sütunları seç ve ilk 3 satırı yazdır
df_new = df[["Rcmnd cruise Knots", "Stall Knots dirty", "Fuel gal/lbs", "Eng out rate of climb", "Price"]]
print(df_new.head(3))
print("")

# Eksik değerlere sahip satırları kaldır
df_new = df_new.dropna()

# Bağımsız ve bağımlı değişkenleri belirle
y = df_new['Price']
x = df_new.drop("Price", axis=1)

"""
Önemli Notlar!!!

Y için normalizasyon yapılmaz
Outliner etkisini azaltır
Model performansını artırır
"""

# Standart ölçekleme uygula
ss = StandardScaler()
x2 = ss.fit_transform(x)

# Ölçeklenmiş veriyi DataFrame olarak yazdır
x2 = pd.DataFrame(x2)
print(x2.head(3))
print("")

# Ölçeklenmiş verilerin ilk sütununun ortalamasını yazdır
print(x2[0].mean())
print("")
