#Ridge regresyon overfitting (Aşırı öğrenme) durumları için kullanılır.

#Ridge regresyon sayesinde bias ve varyans arasındaki dengeyi sağlayabiliriz.

#Ridge regresyonda katsayılar üzerinde regülasyon yapılıyor.

#Ridge regresyonda katsayılar küçülür ama sıfır olmaz. Features öz nitelik azalmaz.

#Ridge regresyon L2 

# y = a1 * x1 + a2 * x2 + ....... + b + alfa * (katsayılar toplamı)**2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge

df = pd.read_csv("dataset/student_scores.csv")

y = df['Scores']
x = df[['Hours']]

plt.style.use("fivethirtyeight")
plt.figure(figsize=(8,8))
plt.scatter(x, y)
plt.show()

lr = LinearRegression()
model = lr.fit(x, y)
print("model skor", model.score(x, y))
print("")


alfalar = [1, 10, 20, 100, 200]

for a in alfalar:
    r = Ridge(alpha = a)
    modelr = r.fit(x,y)
    print("modelr skor", modelr.score(x, y))
    print("modelr katsayı", modelr.coef_)