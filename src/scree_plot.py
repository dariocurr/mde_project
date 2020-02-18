import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib

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

pca = PCA().fit(X)
plt.xticks([x for x in range(0, 11)], [x for x in range(1, 12)])
plt.plot(pca.explained_variance_ratio_)
plt.xlabel('Number of Components')
plt.ylabel('Eigenvalue')
plt.show()

"""
U, S, V = np.linalg.svd(X)

sing_vals = np.arange(len(X[0])) + 1
eigvals = S**2 / np.sum(S**2)
fig = plt.figure(figsize=(8,5))
plt.plot(sing_vals, eigvals, 'ro-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
leg = plt.legend(['Eigenvalues from SVD'], loc='best', borderpad=0.3,
                 shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                 markerscale=0.4)
leg.get_frame().set_alpha(0.4)
plt.show()
"""
