import matplotlib.pyplot as plt
import pandas as pd
import pydotplus
import seaborn as sns
import squarify
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

dataset_path = "../res/dataset_virtuale.csv"


def split_dataset():
    dataset = pd.read_csv(dataset_path, index_col=False)
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
    dotfile = tree.export_graphviz(decision_tree=t,
                                   class_names=["non celiaco", "celiaco"],
                                   feature_names=dataset.columns.drop("Class"),
                                   filled=True, rounded=True)
    graph = pydotplus.graph_from_dot_data(dotfile)
    graph.write_png("../res/decision_tree.png")


def create_treemap(t):
    string = tree.export_text(t, show_weights=True, spacing=1)
    feature_label_stack = list()
    labels = list()
    values = list()
    while string.find("feature") != -1:
        string, feature_label = extract_feature_label(string)
        feature_label_stack = clean_feature_label_stack(feature_label_stack, feature_label)
        if (string.find("weights") < string.find("feature")) or (string.find("feature") == -1):
            label, value = extract_feature_properties(string)
            labels.append(generate_label(feature_label_stack) + feature_label + label)
            values.append(value)
        else:
            feature_label_stack.insert(0, feature_label)
    sorted_labels = list()
    sizes = list()
    while len(labels) != 0:
        maximum = max(values)
        i = len(labels)
        while maximum in values:
            index = values.index(maximum)
            sorted_labels.append(labels[index][labels[index].find("\n") + 1:])
            sizes.append(pow(i, 1.6))
            values.pop(index)
            labels.pop(index)
    palette = sns.light_palette(color=(210, 90, 60),
                                input="husl",
                                n_colors=len(sorted_labels))
    palette.reverse()
    squarify.plot(sizes=sizes,
                  label=sorted_labels,
                  color=palette)
    plt.axis('off')
    plt.show()


def extract_feature_label(string):
    index_feature = string.find("feature")
    string = string[index_feature:]
    space_index = string.find(" ")
    feature_label = "\n" + dataset.columns[int(string[8:space_index])]
    string = string[space_index + 1:]
    line_index = string.find("|")
    feature_label += " " + string[0:line_index - 1]
    string = string[line_index:]
    return string, feature_label


def extract_feature_properties(string):
    open_bracket_index = string.find("[")
    comma_index = string.find(",")
    close_bracket_index = string.find("]")
    num1 = float(string[open_bracket_index + 1:comma_index])
    num2 = float(string[comma_index + 2:close_bracket_index])
    num = int(num1 + num2)
    string = string[close_bracket_index:]
    class_index = string.find("class")
    if string[class_index + 7:class_index + 8] == "0":
        return "\n" + str(num) + " classificati non celiaco/i", num
    else:
        return "\n" + str(num) + " classificati celiaco/i", num


def generate_label(feature_label_stack):
    label = ""
    feature_label_stack.reverse()
    for feature_label in feature_label_stack:
        label += feature_label
    feature_label_stack.reverse()
    return label


def clean_feature_label_stack(feature_label_stack, feature_label):
    opposite_feature_label = ""
    if feature_label.find("<=") != -1:
        opposite_feature_label = feature_label.replace("<=", "> ")
    elif feature_label.find(">") != -1:
        opposite_feature_label = feature_label.replace("> ", "<=")
    if opposite_feature_label in feature_label_stack:
        opposite_feature_label_index = feature_label_stack.index(opposite_feature_label)
        feature_label_stack = feature_label_stack[opposite_feature_label_index + 1:]
    return feature_label_stack

X_train, X_test, Y_train, Y_test, dataset = split_dataset()
decision_tree_classifier(X_train, X_test, Y_train, Y_test, dataset)
