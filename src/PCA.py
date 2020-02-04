"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



# Carico il dataser all'interno di un dataFrame
iris = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                 names=['sepal length','sepal width','petal length','petal width','target'])
#iris.head()


# Creo gli array numpy
X = iris.drop("target", axis=1).values
Y = iris["target"].values


# Codifico i target in numeri
le = LabelEncoder()
Y = le.fit_transform(Y)


# Eseguo la standardizzazione
ss = StandardScaler()
X = ss.fit_transform(X)


# Eseguo la PCA per ridurre lo spazio dimensionale
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)


# Plot 2D
plt.xlabel("Prima componente principale")
plt.ylabel("Seconda componente principale")
plt.scatter(X_pca[:,0], X_pca[:,1], c=Y)


# Plot 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca[:,0], X_pca[:,1], X_pca[:,2], c=Y, marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

dataset = pd.read_csv("dataset_virtuale.csv")
dataset.head()

# Converto i valori NaN con la media
for column in dataset.columns:
    dataset[column] = pd.to_numeric(dataset[column])
    replace_with = dataset[column].mean()
    dataset[column].fillna(replace_with, inplace=True)

dataset.head()

# Split
X = dataset.drop("Class", axis=1)
Y = dataset["Class"]


# Standardizzazione
ss = StandardScaler()
X = ss.fit_transform(X)


# Eseguo la PCA per ridurre lo spazio dimensionale
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

"""
# Plot 2D
plt.xlabel("Prima componente principale")
plt.ylabel("Seconda componente principale")
plt.scatter(X_pca[:,0], X_pca[:,1], c=Y)
"""

# Plot 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca[:,0], X_pca[:,1], X_pca[:,2], c=Y, marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
