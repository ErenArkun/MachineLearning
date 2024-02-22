# MSE RMSE MAE MAPE
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Veri dosyasını oku
df = pd.read_csv("dataset/insurance.csv")

# Veriyi ikili kodlama yaparak dönüştür
df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)

# Bağımlı ve bağımsız değişkenleri tanımla
y = df[['charges']]
x = df.drop('charges', axis = 1)

# Doğrusal regresyon modeli oluştur
lm = LinearRegression()
model = lm.fit(x, y)

# Modelin başarı oranını yazdır
print("Basari Orani:")
print(model.score(x, y))
print("")

# Tahmin yap
tah = model.predict([[19, 26, 0, 1, 1, 0, 0, 1]])
print("Tahmin:")
print(tah)
print("")

# Gerçek ve tahmin edilen değerleri içeren bir veri çerçevesi oluştur
df_hata = pd.DataFrame()

#gerçek sonuç
df_hata['y'] = y
y_tahmin = model.predict(x)

#modelin tahmini
df_hata['tahmin'] = y_tahmin

#gerçek sonuç ve tahimn edilen sonuç farkı
df_hata['hata_payi'] = y-y_tahmin

#eksi sonuçlardan kurtul karelerini al (karesinden kurtulman gerek)
df_hata['tahmin_karesi'] = df_hata['hata_payi']**2
#veya numpy abstract kullan (daha kullanışlı)
df_hata['abs_hata'] = np.abs(df_hata['hata_payi'])

#bu hesaplamada istersen 100 ile çarp
#(gerçek değer - tahmin değer) / gerçek değer
df_hata['percent_error'] = np.abs((y-y_tahmin)/y)
#df_hata['percent_error'] = np.abs((df_hata['y'] - df_hata['tahmin']) / df_hata['y'])


print(df_hata.head(4))
print("")

print(df_hata.mean())
print("")

print(mean_squared_error(y, y_tahmin))
print("")

print(mean_absolute_error(y, y_tahmin))
print("")

print(mean_absolute_percentage_error(y, y_tahmin))
print("")
