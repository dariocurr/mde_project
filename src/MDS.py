import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS


dataset_path = "dataset_virtuale.csv"



def initDataframe():
    # Carico il dataser all'interno di un dataFrame
    dataframe = pd.read_csv(dataset_path)
    #dataframe.head()

    # Converto i valori NaN con la media
    for column in dataframe.columns:
        dataframe[column] = pd.to_numeric(dataframe[column])
        replace_with = dataframe[column].mean()
        dataframe[column].fillna(replace_with, inplace=True)
    #dataframe.head()

    # Creo gli array numpy di addestramento e predizione
    X = dataframe.drop("Class", axis=1).values
    Y = dataframe["Class"].values

    # Eseguo la standardizzazione
    ss = StandardScaler()
    X = ss.fit_transform(X)

    return X, Y



def op1():
    X, Y = initDataframe()

    # Eseguo MDS
    mds = MDS(n_components=2)
    X_mds = mds.fit_transform(X)

    # Plot 2D
    plt.xlabel("Coordinata 1")
    plt.ylabel("Coordinata 2")
    plt.scatter(X_mds[:,0], X_mds[:,1], c=Y)
    plt.show()



def op2():
    X, Y = initDataframe()

    # Eseguo MDS 
    mds = MDS(n_components=3)
    X_mds = mds.fit_transform(X)

    # Plot 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_mds[:,0], X_mds[:,1], X_mds[:,2], c=Y, marker='o')

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
