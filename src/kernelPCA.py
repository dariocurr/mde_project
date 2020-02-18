import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA


dataset_path = "dataset_virtuale.csv"



def initDataframe():
    # Carico il dataser all'interno di un dataFrame
    dataset = pd.read_csv(dataset_path)
    #dataset.head()

    # Converto i valori NaN con la media
    for column in dataset.columns:
        dataset[column] = pd.to_numeric(dataset[column])
        replace_with = dataset[column].mean()
        dataset[column].fillna(replace_with, inplace=True)
    #dataset.head()

    # Split
    X = dataset.drop("Class", axis=1)
    Y = dataset["Class"]

    # Standardizzazione
    ss = StandardScaler()
    X = ss.fit_transform(X)

    return X, Y



def op1():
    X, Y = initDataframe()

    # Eseguo il KernelPCA
    kernelPCA = KernelPCA(n_components=2, kernel="sigmoid")
    X_kernelPCA = kernelPCA.fit_transform(X)

    # Plot 2D
    colormap = np.array(['#ff0000', '#0000ff'])
    plt.scatter(X_kernelPCA[:,0], X_kernelPCA[:,1], c=colormap[Y], alpha=0.5, edgecolors="black", linewidth=1)
    pop_a = mpatches.Patch(color='#ff0000',label='Non Celiaco')
    pop_b = mpatches.Patch(color='#0000ff',label='Celiaco')
    plt.legend(handles=[pop_a,pop_b], loc='upper right')
    plt.title('kernelPCA con n_components = 2')
    plt.xlabel("Prima componente principale")
    plt.ylabel("Seconda componente principale")
    plt.show()



def op2():
    X, Y = initDataframe()

    # Eseguo la t-SNE
    kernelPCA = KernelPCA(n_components=3, kernel='sigmoid')
    X_kernelPCA = kernelPCA.fit_transform(X)

    minimum = min(X_kernelPCA[:,2])
    for i in range(len(X_kernelPCA)):
        jittered_x = np.random.rand() * 2 - 1
        X_kernelPCA[i][0] += jittered_x
        jittered_y = np.random.rand() * 1 - 0.50
        X_kernelPCA[i][1] += jittered_y
        X_kernelPCA[i][2] += abs( 2 * minimum)
        X_kernelPCA[i][2] *= 25

    # Plot 2D
    colormap = np.array(['#ff0000', '#0000ff'])
    plt.scatter(X_kernelPCA[:,0], X_kernelPCA[:,1], s=X_kernelPCA[:,2], c=colormap[Y], alpha=0.5, edgecolors="black", linewidth=1)
    pop_a = mpatches.Patch(color='#ff0000',label='Non Celiaco')
    pop_b = mpatches.Patch(color='#0000ff',label='Celiaco')
    plt.legend(handles=[pop_a,pop_b], loc='upper right')
    plt.title('kernelPCA con n_components = 3')
    plt.xlabel("Coordinata 1")
    plt.ylabel("Coordinata 2")
    plt.show()



def op3():
    X, Y = initDataframe()

    # Eseguo il kernelPCA
    kernelPCA = KernelPCA(n_components=3, kernel="sigmoid")
    X_kernelPCA = kernelPCA.fit_transform(X)

    # Plot 3D
    colormap = np.array(['#ff0000', '#0000ff'])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_kernelPCA[:,0], X_kernelPCA[:,1], X_kernelPCA[:,2], c=colormap[Y], marker='o', alpha=0.5, edgecolors="black", linewidth=1)
    pop_a = mpatches.Patch(color='#ff0000',label='Non Celiaco')
    pop_b = mpatches.Patch(color='#0000ff',label='Celiaco')
    plt.legend(handles=[pop_a,pop_b], loc='upper right')
    plt.title('kernelPCA con n_components = 3')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()



def menu():
    os.system("clear")

    print("1. 2D (n_components = 2)")
    print("2. 2D (n_components = 3)")
    print("3. interactive 3D")
    print("0. Exit\n")

    temp = int(input())
    print("\n")
    return temp



c = -1
while c != 0:
    c = menu()
    if c == 1:
        op1()
    elif c == 2:
        op2()
    elif c == 3:
        op3()
    elif c == 0:
        break
    else:
        print("Unknown command")
