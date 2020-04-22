import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
pd.set_option('display.width', 999)


dataset_path = "../res/dataset_virtuale.csv"


def initDataframe():
    # Carico il dataser all'interno di un dataFrame
    dataset = pd.read_csv(dataset_path, index_col=False)
    datasetWithNan = pd.read_csv(dataset_path)
    # Converto i valori NaN con la media
    for column in dataset.columns:
        dataset[column] = pd.to_numeric(dataset[column])
        replace_with = dataset[column].mean()
        dataset[column].fillna(replace_with, inplace=True)
    # Split
    X = dataset.drop("Class", axis=1)
    Y = dataset["Class"]
    # Standardizzazione
    ss = StandardScaler()
    X = ss.fit_transform(X)
    return X, Y, datasetWithNan


def PCA_2D():
    # Eseguo la PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    """
    # Ricerca dei data point appartenenti al cluster lineare centrale
    listFilter = []
    dataset_np = datasetWithNan.to_numpy()
    for i in range(X_pca.shape[0]):
        if(X_pca[i][0] > 7 and X_pca[i][0] < 10):
            listFilter.append(dataset_np[i])
    dataFilter_np = np.array(listFilter)
    pd.DataFrame(data=dataFilter_np, columns=['Anemia', 'Osteopenia', 'Diarrea Cronica', 'Mancata Crescita', 'Disturbi Genetici',
                                              'Madre Celiaca', 'POCT', 'IGA_totali', 'TTG_IGG', 'TTG_IGA', 'Esami del sangue', 'Class']).to_csv("dataset_filtrato.csv", index=False)
    """

    # Plot 2D
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colormap[Y], alpha=alpha, edgecolors=edgecolors, linewidth=linewidth)
    plt.legend(handles=[pop_a, pop_b], loc='upper right')
    plt.title("PCA con n = 2 componenti")
    plt.xlabel("Prima componente principale")
    plt.ylabel("Seconda componente principale")
    plt.show()
    plt.savefig("../Articolo/img/PCA_2D.png")


def PCA_2D_with_size():
    # Eseguo la PCA
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)
    maximum = max(X_pca[:, 2])
    for i in range(len(X_pca)):
        jittered_x = np.random.rand() * 2 - 1
        X_pca[i][0] += jittered_x
        jittered_y = np.random.rand() * 1 - 0.50
        X_pca[i][1] += jittered_y
        X_pca[i][2] += abs(2 * min(X_pca[:, 2]))
        X_pca[i][2] *= (300 / maximum)
    # Plot 2D
    plt.scatter(X_pca[:, 0], X_pca[:, 1], s=X_pca[:, 2], c=colormap[Y],
                alpha=alpha, edgecolors=edgecolors, linewidth=linewidth)
    plt.legend(handles=[pop_a, pop_b], loc='upper right')
    plt.title("PCA con n = 3 componenti")
    plt.xlabel("Prima componente principale")
    plt.ylabel("Seconda componente principale")
    plt.show()
    plt.savefig("../Articolo/img/PCA_2D_with_size.png")


def PCA_i3D():
    # Eseguo la PCA
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)
    # Plot 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=colormap[Y], marker='o',
               alpha=alpha, edgecolors=edgecolors, linewidth=linewidth)
    plt.legend(handles=[pop_a, pop_b], loc='upper right')
    plt.title("PCA con n = 3 componenti")
    ax.set_xlabel("Prima componente principale")
    ax.set_ylabel("Seconda componente principale")
    ax.set_zlabel("Terza componente principale")
    plt.show()
    plt.savefig("../Articolo/img/PCA_i3D.png")


def PCA_SPLOM(n_params):
    # Eseguo la PCA
    pca = PCA(n_components=n_params)
    X_pca = pca.fit_transform(X)
    # Creo il dizionario dei componenti
    components = {}
    for i in range(n_params):
        components.update({'Componente{0}'.format(i + 1): X_pca[:, i]})
    # Converto il numpy array in un dataframe
    dataframe = pd.DataFrame(components)
    dataframe["Class"] = Y
    dataframe["Class"].replace(to_replace=0, value="Non Celiaco", inplace=True)
    dataframe["Class"].replace(to_replace=1, value="Celiaco", inplace=True)
    # SPLOM plot
    sns.set(style="ticks")
    sns.pairplot(dataframe, hue="Class", palette={"Non Celiaco": blue, "Celiaco": yellow}, plot_kws=dict(
        edgecolor=edgecolors, linewidth=linewidth, alpha=alpha))
    plt.show()
    plt.savefig("../Articolo/img/PCA_SPLOM.png")


def PCA_components_interpretation():
    pca = PCA().fit(X)
    components = pd.DataFrame(pca.components_)
    features = ["Anemia", "Osteopenia", "Diarrea Cronica", "Mancata Crescita", "Disturbi Genetici",
                "Madre Celiaca", "POCT", "IGA_totali", "TTG_IGG", "TTG_IGA", "Esami del sangue"]
    columns = dict()
    rows = dict()
    for i in range(0, len(components)):
        rows[i] = "PCA" + str(i + 1)
        columns[i] = features[i]
    components.rename(index=rows, columns=columns, inplace=True)
    print(pd.DataFrame(components.transpose()))


def PCA_generate_file(n_params):
    pca = PCA(n_components=n_params)
    X_PCA = pca.fit_transform(X)
    X_PCA = pd.DataFrame(data=X_PCA, columns=["Componente" + str(i) for i in range(1, len(X_PCA[0]) + 1)])
    X_PCA["Class"] = Y
    X_PCA.to_csv("../res/dataset_virtuale_PCA.csv", index=False)


def kernelPCA_2D():
    # Eseguo il KernelPCA
    kernelPCA = KernelPCA(n_components=2, kernel="sigmoid")
    X_kernelPCA = kernelPCA.fit_transform(X)
    # Plot 2D
    plt.scatter(X_kernelPCA[:, 0], X_kernelPCA[:, 1], c=colormap[Y],
                alpha=alpha, edgecolors=edgecolors, linewidth=linewidth)
    plt.legend(handles=[pop_a, pop_b], loc='upper right')
    plt.title("kernelPCA con n = 2 componenti")
    plt.xlabel("Prima componente principale")
    plt.ylabel("Seconda componente principale")
    plt.show()
    plt.savefig("../Articolo/img/kernelPCA_2D.png")


def kernelPCA_2D_with_size():
    # Eseguo la t-SNE
    kernelPCA = KernelPCA(n_components=3, kernel='sigmoid')
    X_kernelPCA = kernelPCA.fit_transform(X)
    minimum = min(X_kernelPCA[:, 2])
    maximum = max(X_kernelPCA[:, 2])
    for i in range(len(X_kernelPCA)):
        jittered_x = np.random.rand()
        X_kernelPCA[i][0] += jittered_x
        jittered_y = np.random.rand()
        X_kernelPCA[i][1] += jittered_y
        X_kernelPCA[i][2] += abs(2 * minimum)
        X_kernelPCA[i][2] *= (300 / maximum)
    # Plot 2D
    plt.scatter(X_kernelPCA[:, 0], X_kernelPCA[:, 1], s=X_kernelPCA[:, 2],
                c=colormap[Y], alpha=alpha, edgecolors=edgecolors, linewidth=linewidth)
    plt.legend(handles=[pop_a, pop_b], loc='upper right')
    plt.title("kernelPCA con n = 3 componenti")
    plt.xlabel("Prima componente principale")
    plt.ylabel("Seconda componente principale")
    plt.show()
    plt.savefig("../Articolo/img/kernelPCA_2D_with_size.png")


def kernelPCA_i3D():
    # Eseguo il kernelPCA
    kernelPCA = KernelPCA(n_components=3, kernel="sigmoid")
    X_kernelPCA = kernelPCA.fit_transform(X)
    # Plot 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_kernelPCA[:, 0], X_kernelPCA[:, 1], X_kernelPCA[:, 2], c=colormap[Y],
               marker='o', alpha=alpha, edgecolors=edgecolors, linewidth=linewidth)
    plt.legend(handles=[pop_a, pop_b], loc='upper right')
    plt.title("kernelPCA con n = 3 componenti")
    ax.set_xlabel("Prima componente principale")
    ax.set_ylabel("Seconda componente principale")
    ax.set_zlabel("Terza componente principale")
    plt.show()
    plt.savefig("../Articolo/img/kernelPCA_i3D.png")


def kernelPCA_SPLOM(n_params):
    # Eseguo la kernelPCA
    kernelPCA = KernelPCA(n_components=n_params)
    X_kernelPCA = kernelPCA.fit_transform(X)
    # Creo il dizionario dei componenti
    components = {}
    for i in range(n_params):
        components.update({'Componente{0}'.format(i + 1): X_kernelPCA[:, i]})
    # Converto il numpy array in un dataframe
    dataframe = pd.DataFrame(components)
    dataframe["Class"] = Y
    dataframe["Class"].replace(to_replace=0, value="Non Celiaco", inplace=True)
    dataframe["Class"].replace(to_replace=1, value="Celiaco", inplace=True)
    # SPLOM plot
    sns.set(style="ticks")
    sns.pairplot(dataframe, hue="Class", palette={"Non Celiaco": blue, "Celiaco": yellow}, plot_kws=dict(
        edgecolor=edgecolors, linewidth=linewidth, alpha=alpha))
    plt.show()
    plt.savefig("../Articolo/img/kernelPCA_SPLOM.png")


def kernelPCA_generate_file(n_params):
    kernelPCA = KernelPCA(n_components=n_params, kernel="sigmoid")
    X_kernelPCA = kernelPCA.fit_transform(X)
    X_kernelPCA = pd.DataFrame(data=X_kernelPCA, columns=["Componente" + str(i)
                                                          for i in range(1, len(X_kernelPCA[0]) + 1)])
    X_kernelPCA["Class"] = Y
    X_kernelPCA.to_csv("../res/dataset_virtuale_kernelPCA.csv", index=False)


def MDS_2D():
    # Eseguo MDS
    mds = MDS(n_components=2)
    X_mds = mds.fit_transform(X)
    # Plot 2D
    plt.scatter(X_mds[:, 0], X_mds[:, 1], c=colormap[Y], alpha=alpha, edgecolors=edgecolors, linewidth=linewidth)
    plt.legend(handles=[pop_a, pop_b], loc='upper right')
    plt.title("MDS con n = 2 componenti")
    plt.xlabel("Coordinata X")
    plt.ylabel("Coordinata Y")
    plt.show()
    plt.savefig("../Articolo/img/MDS_2D.png")


def MDS_2D_with_size():
    # Eseguo la MDS
    mds = MDS(n_components=3)
    X_mds = mds.fit_transform(X)
    minimum = min(X_mds[:, 2])
    maximum = max(X_mds[:, 2])
    for i in range(len(X_mds)):
        jittered_x = np.random.rand()
        X_mds[i][0] += jittered_x
        jittered_y = np.random.rand()
        X_mds[i][1] += jittered_y
        X_mds[i][2] += abs(2 * minimum)
        X_mds[i][2] *= (300 / maximum)
    # Plot 2D
    plt.scatter(X_mds[:, 0], X_mds[:, 1], s=X_mds[:, 2], c=colormap[Y],
                alpha=0.5, edgecolors=edgecolors, linewidth=linewidth)
    plt.legend(handles=[pop_a, pop_b], loc='upper right')
    plt.title("MDS con n = 3 componenti")
    plt.xlabel("Coordinata X")
    plt.ylabel("Coordinata Y")
    plt.show()
    plt.savefig("../Articolo/img/MDS_2D_with_size.png")


def MDS_i3D():
    # Eseguo MDS
    mds = MDS(n_components=3)
    X_mds = mds.fit_transform(X)
    # Plot 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_mds[:, 0], X_mds[:, 1], X_mds[:, 2], c=colormap[Y], marker='o',
               alpha=alpha, edgecolors=edgecolors, linewidth=linewidth)
    plt.legend(handles=[pop_a, pop_b], loc='upper right')
    plt.title("MDS con n = 3 componenti")
    ax.set_xlabel("Coordinata X")
    ax.set_ylabel("Coordinata Y")
    ax.set_zlabel("Coordinata Z")
    plt.show()
    plt.savefig("../Articolo/img/MDS_i3D.png")


def MDS_SPLOM(n_params):
    # Eseguo la MDS
    mds = MDS(n_components=n_params)
    X_mds = mds.fit_transform(X)
    # Creo il dizionario dei componenti
    components = {}
    for i in range(n_params):
        components.update({'Componente{0}'.format(i + 1): X_mds[:, i]})
    # Converto il numpy array in un dataframe
    dataframe = pd.DataFrame(components)
    dataframe["Class"] = Y
    dataframe["Class"].replace(to_replace=0, value="Non Celiaco", inplace=True)
    dataframe["Class"].replace(to_replace=1, value="Celiaco", inplace=True)
    # SPLOM plot
    sns.set(style="ticks")
    sns.pairplot(dataframe, hue="Class", palette={"Non Celiaco": blue, "Celiaco": yellow}, plot_kws=dict(
        edgecolor=edgecolors, linewidth=linewidth, alpha=alpha))
    plt.show()
    plt.savefig("../Articolo/img/MDS_SPLOM.png")


def MDS_generate_file(n_params):
    mds = MDS(n_components=3)
    X_MDS = mds.fit_transform(X)
    X_MDS = pd.DataFrame(data=X_MDS, columns=["Componente" + str(i) for i in range(1, len(X_MDS[0]) + 1)])
    X_MDS["Class"] = Y
    X_MDS.to_csv("../res/dataset_virtuale_MDS.csv", index=False)


def tSNE_2D():
    # Eseguo il t-SNE
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(X)
    # Plot 2D
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=colormap[Y], alpha=alpha, edgecolors=edgecolors, linewidth=linewidth)
    plt.legend(handles=[pop_a, pop_b], loc='upper right')
    plt.title("t-SNE con n = 2 componenti")
    plt.xlabel("Prima Componente")
    plt.ylabel("Seconda Componente")
    plt.show()
    plt.savefig("../Articolo/img/tSNE_2D.png")


def tSNE_2D_with_size():
    # Eseguo la t-SNE
    tsne = TSNE(n_components=3)
    X_tsne = tsne.fit_transform(X)
    minimum = min(X_tsne[:, 2])
    maximum = max(X_tsne[:, 2])
    for i in range(len(X_tsne)):
        jittered_x = np.random.rand()
        X_tsne[i][0] += jittered_x
        jittered_y = np.random.rand()
        X_tsne[i][1] += jittered_y
        X_tsne[i][2] += abs(2 * minimum)
        X_tsne[i][2] *= (300 / maximum)
    # Plot 2D
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=X_tsne[:, 2], c=colormap[Y],
                alpha=alpha, edgecolors=edgecolors, linewidth=linewidth)
    plt.legend(handles=[pop_a, pop_b], loc='upper right')
    plt.title("t-SNE con n = 3 componenti")
    plt.xlabel("Prima Componente")
    plt.ylabel("Seconda Componente")
    plt.show()
    plt.savefig("../Articolo/img/tSNE_2D_with_size.png")


def tSNE_i3D():
    # Eseguo il t-SNE
    tsne = TSNE(n_components=3)
    X_tsne = tsne.fit_transform(X)
    # Plot 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], c=colormap[Y],
               marker='o', alpha=alpha, edgecolors=edgecolors, linewidth=linewidth)
    plt.legend(handles=[pop_a, pop_b], loc='upper right')
    plt.title("t-SNE con n = 3 componenti")
    ax.set_xlabel("Prima Componente")
    ax.set_ylabel("Seconda Componente")
    ax.set_zlabel("Terza Componente")
    plt.show()
    plt.savefig("../Articolo/img/tSNE_i3D.png")


def tSNE_SPLOM(n_params):
    # Eseguo la t-SNE
    tsne = TSNE(n_components=n_params)
    X_tsne = tsne.fit_transform(X)
    # Creo il dizionario dei componenti
    components = {}
    for i in range(n_params):
        components.update({'Componente{0}'.format(i + 1): X_tsne[:, i]})
    # Converto il numpy array in un dataframe
    dataframe = pd.DataFrame(components)
    dataframe["Class"] = Y
    dataframe["Class"].replace(to_replace=0, value="Non Celiaco", inplace=True)
    dataframe["Class"].replace(to_replace=1, value="Celiaco", inplace=True)
    # SPLOM plot
    sns.set(style="ticks")
    sns.pairplot(dataframe, hue="Class", palette={"Non Celiaco": blue, "Celiaco": yellow}, plot_kws=dict(
        edgecolor=edgecolors, linewidth=linewidth, alpha=alpha))
    plt.show()
    plt.savefig("../Articolo/img/tSNE_SPLOM.png")


def tSNE_generate_file(n_params):
    tsne = TSNE(n_components=n_params)
    X_tSNE = tsne.fit_transform(X)
    X_tSNE = pd.DataFrame(data=X_tSNE, columns=["Componente" + str(i) for i in range(1, len(X_tSNE[0]) + 1)])
    X_tSNE["Class"] = Y
    X_tSNE.to_csv("../res/dataset_virtuale_tSNE.csv", index=False)


def scree_plot():
    pca = PCA().fit(X)
    plt.xticks([x for x in range(0, 11)], [x for x in range(1, 12)])
    plt.plot(pca.explained_variance_ratio_)
    plt.xlabel("Numero di componenti")
    plt.ylabel("Autovalore")
    plt.show()
    plt.savefig("../Articolo/img/scree_plot.png")


def menu():
    os.system("clear")
    print("1. PCA")
    print("2. kernel PCA")
    print("3. MDS")
    print("4. t-SNE")
    print("5. Scree-Plot")
    print("0. Exit\n")
    temp = int(input())
    print()
    return temp


def sub_menu():
    os.system("clear")
    print("1. 2D (2 dimensioni)")
    print("2. 2D (3 dimensioni)")
    print("3. i3D")
    print("4. plot di Draftman")
    print("5. generazione dataset ")
    print("0. Exit\n")
    temp = int(input())
    print()
    return temp


blue = "#0000ff"
yellow = "#ffff00"
X, Y, datasetWithNan = initDataframe()
colormap = np.array([blue, yellow])
alpha = 0.5
edgecolors = "black"
pop_a = mpatches.Patch(color=blue, label='Non Celiaco')
pop_b = mpatches.Patch(color=yellow, label='Celiaco')
linewidth = 1
c = -1
while c != 0:
    c = menu()
    if c == 0:
        break
    elif c == 5:
        scree_plot()
    elif (c > 0) and (c < 5):
        sc = sub_menu()
        if c == 1:
            PCA_components_interpretation()
            if sc == 1:
                PCA_2D()
            elif sc == 2:
                PCA_2D_with_size()
            elif sc == 3:
                PCA_i3D()
            elif sc == 4:
                n_params = -1
                while(n_params <= 1 or n_params > 12):
                    print("Numero di parametri: ", end="")
                    n_params = int(input())
                    if(n_params <= 1 or n_params > 12):
                        print("Numero di parametri non ammesso\n")
                PCA_SPLOM(n_params)
            elif sc == 5:
                n_params = -1
                while(n_params <= 1 or n_params > 12):
                    print("Numero di parametri: ", end="")
                    n_params = int(input())
                    if(n_params <= 1 or n_params > 12):
                        print("Numero di parametri non ammesso\n")
                PCA_generate_file(n_params)
            elif c == 0:
                break
            else:
                print("Comando sconosciuto")
        elif c == 2:
            if sc == 1:
                kernelPCA_2D()
            elif sc == 2:
                kernelPCA_2D_with_size()
            elif sc == 3:
                kernelPCA_i3D()
            elif sc == 4:
                n_params = -1
                while(n_params <= 1 or n_params > 12):
                    print("Numero di parametri: ", end="")
                    n_params = int(input())
                    if(n_params <= 1 or n_params > 12):
                        print("Numero di parametri non ammesso\n")
                kernelPCA_SPLOM(n_params)
            elif sc == 5:
                n_params = -1
                while(n_params <= 1 or n_params > 12):
                    print("Numero di parametri: ", end="")
                    n_params = int(input())
                    if(n_params <= 1 or n_params > 12):
                        print("Numero di parametri non ammesso\n")
                kernelPCA_generate_file(n_params)
            elif c == 0:
                break
            else:
                print("Comando sconosciuto")
        elif c == 3:
            if sc == 1:
                MDS_2D()
            elif sc == 2:
                MDS_2D_with_size()
            elif sc == 3:
                MDS_i3D()
            elif sc == 4:
                n_params = -1
                while(n_params <= 1 or n_params > 12):
                    print("Numero di parametri: ", end="")
                    n_params = int(input())
                    if(n_params <= 1 or n_params > 12):
                        print("Numero di parametri non ammesso\n")
                MDS_SPLOM(n_params)
            elif sc == 5:
                n_params = -1
                while(n_params <= 1 or n_params > 12):
                    print("Numero di parametri: ", end="")
                    n_params = int(input())
                    if(n_params <= 1 or n_params > 12):
                        print("Numero di parametri non ammesso\n")
                MDS_generate_file(n_params)
            elif c == 0:
                break
            else:
                print("Comando sconosciuto")
        elif c == 4:
            if sc == 1:
                tSNE_2D()
            elif sc == 2:
                tSNE_2D_with_size()
            elif sc == 3:
                tSNE_i3D()
            elif sc == 4:
                n_params = -1
                while(n_params <= 1 or n_params > 3):
                    print("Numero di parametri: ", end="")
                    n_params = int(input())
                    if(n_params <= 1 or n_params > 3):
                        print("Numero di parametri non ammesso\n")
                tSNE_SPLOM(n_params)
            elif sc == 5:
                n_params = -1
                while(n_params <= 1 or n_params > 3):
                    print("Numero di parametri: ", end="")
                    n_params = int(input())
                    if(n_params <= 1 or n_params > 3):
                        print("Numero di parametri non ammesso\n")
                tSNE_generate_file(n_params)
            elif c == 0:
                break
            else:
                print("Comando sconosciuto")
    else:
        print("Comando sconosciuto")
