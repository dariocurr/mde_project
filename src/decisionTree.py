import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from subprocess import call
import pydotplus
import matplotlib.pyplot as plt
import os
import collections
path="dataset_virtuale.csv"


def split_dataset():
    dataset=pd.read_csv(path)
    for c in dataset.columns:
        dataset[c]=pd.to_numeric(dataset[c])
        replace_with = dataset[c].mean()
        dataset[c].fillna(replace_with, inplace=True)
    X = dataset.drop("Class", axis=1)
    Y = dataset["Class"]
    ss=StandardScaler()
    X=ss.fit_transform(X)
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)
    return X_train,X_test,Y_train,Y_test,dataset



def decisionTreeClassifier(X_train,X_test,Y_train,Y_test,d,n,m,s):
    os.chdir("img")
    p=np.arange(n,m+2,2)
    n=p.shape
    acc=np.zeros((2,n[0]))
    x=0
    c=["0","1"]
    for i in p:
        t=DecisionTreeClassifier(max_depth=i,criterion=s,splitter="random")
        t.fit(X_train,Y_train)
        Y_pred_train=t.predict(X_train)
        Y_pred_test=t.predict(X_test)
        acc_train=accuracy_score(Y_train,Y_pred_train)
        acc[0][x]=acc_train
        acc_test=accuracy_score(Y_test,Y_pred_test)
        acc[1][x]=acc_test
        print("ACCURACY: TRAIN=%.4f TEST=%.4f" % (acc_train,acc_test))
        title="tree_depth" +str(i)
        title_png="tree_depth" +str(i)+".png"
        dotfile=export_graphviz(t, out_file = None, class_names=c,feature_names = d.columns.drop("Class"),filled=True,rounded=True)
        graph = pydotplus.graph_from_dot_data(dotfile)
        colors = ('turquoise', 'orange')
        edges = collections.defaultdict(list)

        for edge in graph.get_edge_list():
            edges[edge.get_source()].append(int(edge.get_destination()))
            for edge in edges:
                edges[edge].sort()
                for i in range(1):
                    dest = graph.get_node(str(edges[edge][i]))[0]
                    dest.set_fillcolor(colors[i])
        graph.write_png(title_png)
        x=x+1

    plt.figure(figsize=(12,6))
    plt.plot(p,acc[0],color="blue",marker='o',linewidth=2,label="Training Set")
    plt.plot(p,acc[1],color="red",marker='o',linewidth=2,label="Testing Set")
    plt.xlabel("Profondtà massima dell'albero")
    plt.ylabel("ACCURACY")
    plt.title("ANDAMENTO DELL'ACCURACY AL VARIARE DELLA PROFONDITÀ DELL'ALBERO")
    plt.legend()
    plt.savefig("andamento_acc")
    plt.show()





X_train,X_test,Y_train,Y_test,d=split_dataset()
print("Min depth--->")
n=input()
n=int(n)
print("Max depth--->")
m=input()
m=int(m)
print("Criterion--->")
s=input()
decisionTreeClassifier(X_train,X_test,Y_train,Y_test,d,n,m,s)
