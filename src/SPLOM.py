import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataframe=pd.read_csv("dataset_virtuale.csv")

for column in dataframe.columns:
    dataframe[column] = pd.to_numeric(dataframe[column])
    replace_with = dataframe[column].mean()
    dataframe[column].fillna(replace_with, inplace=True)

fig = plt.figure()
sns.set(style="ticks")
sns.pairplot(dataframe.drop("Class",axis=1))
plt.show()
