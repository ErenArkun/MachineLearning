import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz #bilgisayarda kurulu olması gerek. sistem değişkenlerine bin klasörünü path ekle.
from sklearn.model_selection import train_test_split

df = pd.read_csv("dataset/heart.csv")


#version 1 (overfitting) ezberlemiş
"""

y = df["output"]
x = df.drop("output", axis=1)

tree = DecisionTreeClassifier()
model = tree.fit(x, y)
skr = model.score(x, y)
print(skr)

"""

#version 2

y = df["output"]
x = df.drop("output", axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=16, train_size=0.7)

tree = DecisionTreeClassifier()
model = tree.fit(x_train, y_train)
skr = model.score(x_test, y_test)
print(skr)


#veriyi görselleştir
dot = export_graphviz(model, feature_names=x.columns, filled=True)
gorsel = graphviz.Source(dot)
gorsel.render("Classifier/DecisionTree/karar.gv", view=True)
