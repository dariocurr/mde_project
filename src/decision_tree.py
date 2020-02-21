import os
import graphviz
import squarify
import pydotplus
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
dataset_path = "dataset_virtuale.csv"


def split_dataset():
    dataset = pd.read_csv(dataset_path)
    for column in dataset.columns:
        dataset[column] = pd.to_numeric(dataset[column])
        replace_with = dataset[column].mean()
        dataset[column].fillna(replace_with, inplace=True)
    X = dataset.drop("Class", axis=1)
    Y = dataset["Class"].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
    return X_train, X_test, Y_train, Y_test, dataset


def decision_tree_classifier(X_train, X_test, Y_train, Y_test, dataset):
    decision_tree = tree.DecisionTreeClassifier(splitter="random")
    decision_tree.fit(X_train, Y_train)
    Y_pred_train = decision_tree.predict(X_train)
    Y_pred_test = decision_tree.predict(X_test)
    acc_train = accuracy_score(Y_train, Y_pred_train)
    acc_test = accuracy_score(Y_test, Y_pred_test)
    print("ACCURACY: TRAIN=%.4f TEST=%.4f" % (acc_train, acc_test))
    create_binary_tree(decision_tree)
    create_treemap(decision_tree)


def create_binary_tree(t):
    dotfile = tree.export_graphviz(decision_tree=t, class_names=[
                                   "non celiaco", "celiaco"], feature_names=dataset.columns.drop("Class"), filled=True, rounded=True)
    graph = pydotplus.graph_from_dot_data(dotfile)
    graph.write_png("../res/tree_depth.png")


def create_treemap(t):
    data = tree.export_text(t, show_weights=True, spacing=1)
    labels = list()
    values = list()
    while data.find("feature") != -1:
        index_feature = data.find("feature")
        data = data[index_feature:]
        space_index = data.find(" ")
        label = dataset.columns[int(data[8:space_index])]
        data = data[space_index + 1:]
        line_index = data.find("|")
        label += " " + data[0:line_index - 1]
        data = data[line_index:]
        if(data.find("weights") < data.find("feature") or data.find("feature") == -1):
            open_bracket_index = data.find("[")
            comma_index = data.find(",")
            close_bracket_index = data.find("]")
            num1 = float(data[open_bracket_index + 1:comma_index])
            num2 = float(data[comma_index + 2:close_bracket_index])
            labels.append(label)
            values.append(int(num1 + num2))
    print(str(labels) + " " + str(values))
    squarify.plot(sizes=values, label=labels, alpha=0.7)
    plt.axis('off')
    plt.show()


X_train, X_test, Y_train, Y_test, dataset = split_dataset()
decision_tree_classifier(X_train, X_test, Y_train, Y_test, dataset)
