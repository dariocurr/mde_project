import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
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
    plt.xlabel("Prima componente principale")
    plt.ylabel("Seconda componente principale")
    plt.scatter(X_kernelPCA[:,0], X_kernelPCA[:,1], c=Y)
    plt.show()



def op2():
    X, Y = initDataframe()

    # Eseguo il kernelPCA
    kernelPCA = KernelPCA(n_components=3, kernel="sigmoid")
    X_kernelPCA = kernelPCA.fit_transform(X)

    # Plot 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_kernelPCA[:,0], X_kernelPCA[:,1], X_kernelPCA[:,2], c=Y, marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()



def menu():
    os.system("clear")

    print("1. Plot 2D")
    print("2. Plot 3D")
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
    elif c == 0:
        break
    else:
        print("Unknown command")
        os.system("clear")
