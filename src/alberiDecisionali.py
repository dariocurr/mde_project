import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz



# Creazione dataframe
dataframe = pd.read_csv("dataset_virtuale.csv")
dataframe.head()


# Conversione dei valori NaN con la media
for column in dataframe.columns:
    dataframe[column] = pd.to_numeric(dataframe[column])
    replace_with = dataframe[column].mean()
    dataframe[column].fillna(replace_with, inplace=True)
dataframe.head()


# Split
X = dataframe.drop("Class", axis=1)
Y = dataframe["Class"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0)
X_train.shape
X_test.shape


# Albero decisionale
tree = DecisionTreeClassifier(criterion="gini")
tree.fit(X_train, Y_train)

y_pred_train = tree.predict(X_train)
y_pred = tree.predict(X_test)

accuracy_train = accuracy_score(Y_train, y_pred_train)
accuracy_test = accuracy_score(Y_test, y_pred)

print("ACCURACY: TRAIN=%.4f TEST=%.4f" % (accuracy_train, accuracy_test))

dotfile = open("tree.dot", "w")
export_graphviz(tree, out_file=dotfile, feature_names=dataframe.columns.drop("Class"))
dotfile.close()
