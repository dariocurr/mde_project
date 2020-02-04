import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS


# Carico il dataser all'interno di un dataFrame
dataframe = pd.read_csv("dataset_virtuale.csv")

# Converto i valori NaN con la media
for column in dataframe.columns:
    dataframe[column] = pd.to_numeric(dataframe[column])
    replace_with = dataframe[column].mean()
    dataframe[column].fillna(replace_with, inplace=True)

# Creo gli array numpy di addestramento e predizione
X = dataframe.drop("Class", axis=1).values
Y = dataframe["Class"].values

# Eseguo la standardizzazione
ss = StandardScaler()
X = ss.fit_transform(X)

# Eseguo la PCA per ridurre lo spazio dimensionale
mds = MDS(n_components=3)
X_mds = mds.fit_transform(X)

"""
# Plot 2D
plt.xlabel("Coordinata 1")
plt.ylabel("Coordinata 2")
plt.scatter(X_mds[:,0], X_mds[:,1], c=Y)
plt.show()
"""

# Plot 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_mds[:, 0], X_mds[:, 1], X_mds[:, 2], c=Y, marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
