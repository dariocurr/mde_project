import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE



# Carico il dataser all'interno di un dataFrame
dataframe = pd.read_csv("dataset_virtuale.csv")
#dataframe.head()


# Converto i valori NaN con la media
for column in dataframe.columns:
    dataframe[column] = pd.to_numeric(dataframe[column])
    replace_with = dataframe[column].mean()
    dataframe[column].fillna(replace_with, inplace=True)
#dataframe.head()


# Creo gli array numpy
X = dataframe.drop("Class", axis=1).values
Y = dataframe["Class"].values


# Eseguo la standardizzazione
ss = StandardScaler()
X = ss.fit_transform(X)


# Eseguo la PCA per ridurre lo spazio dimensionale
tsne = TSNE(n_components=3)
X_tsne = tsne.fit_transform(X)


"""
# Plot 2D
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=Y)
"""


# Plot 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_tsne[:,0], X_tsne[:,1], X_tsne[:,2], c=Y, marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
