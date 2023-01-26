#!/usr/bin/env python3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv("/home/ubuntu/data/heart_2020_cleaned.csv")
df=data.copy()
print('-------basic information of each column-------')
print(df.info())
df_cat=df.select_dtypes(include=["object"])
print('-------first 10 rows of cat vars-------')
print(df_cat.head(10))
print('-------examine the distribution of cat vars-------')
def frequencies(variable):
    """
    input: variable Ex:"GenHealth","AgeCategory" ...
    output: Bar plot & value count
    """
    # Get columns
    var = df_cat[variable]
    # Frequencies of categorical variables
    varValue = var.value_counts()
    # Visualization
    plt.bar(varValue.index, varValue)
    plt.xticks(varValue.index, varValue.index.values)
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()
    print("{}: \n {}".format(variable, varValue))
for i in df_cat:
    frequencies(i)
df_num=df.select_dtypes(include=["float64"])
print('-------first 10 rows of cat vars-------')
print(df_num.head(10))
print('-------statistical summary for num vars-------')
print(df_num.describe().T)
print('-------use boxplot to examine the distribution of num vars-------')
plt.figure(figsize=(10,8))
sns.set_theme(style="whitegrid")
sns.boxplot(data=df_num, showfliers=False)
plt.xlabel("Numerical Attributes", fontsize= 12)
plt.ylabel("Values", fontsize= 12)
plt.title("Numerical Attributes Boxplot", fontsize= 15)
plt.show()
