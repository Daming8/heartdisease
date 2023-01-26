#!/usr/bin/env python3
import pandas as pd
import seaborn as sns
import numpy as np

data = pd.read_csv("/home/ubuntu/data/heart_2020_cleaned.csv")
df=data.copy()
heartDisease="HeartDisease"
df_num=df.select_dtypes(include=["float64"])
df_num_sleepTime=df_num["SleepTime"]
sns.boxplot(x=df_num_sleepTime);


Q1=df_num_sleepTime.quantile(0.25)
Q3=df_num_sleepTime.quantile(0.75)
IQR=Q3-Q1
print("First Quartile: ",Q1)
print("Third Quartile: ",Q3)
print("Interquartile: ",IQR)
lower_limit=Q1-1.5*IQR
upper_limit=Q3+1.5*IQR
print("Lower Limit: ",lower_limit)
print("Upper Limi: ",upper_limit)
outlier_TF=(df_num_sleepTime<lower_limit)
df_num_sleepTime=pd.DataFrame(df_num_sleepTime)
sleepTime_df=df_num_sleepTime[~((df_num_sleepTime<(lower_limit)) | (df_num_sleepTime>(upper_limit))).any(axis=1)]
df_num_sleepTime[outlier_TF]=df_num_sleepTime.mean()
df["id"]=range(1,319796)
df=df.set_index("id")
df_cat=df.select_dtypes(include=["object"])
df_categorical=df_cat.columns
for var in df_categorical:
    Dummy=pd.get_dummies(df[var]).add_prefix(var)
    df=df.merge(Dummy,on="id")
    df=df.drop([var],axis=1)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
columns=df.columns
s=scaler.fit_transform(df)

scaled_data=pd.DataFrame(s,columns=columns)
print(scaled_data.head(5))
scaled_data.to_csv('/home/ubuntu/data/heart_scaled.csv')
result=pd.DataFrame({'Model':[],'Accuracy':[],'Precision':[],'Recall':[],'F1':[]})
result.to_csv('/home/ubuntu/data/result.csv',header=['Model','Accuracy','Precision','Recall','F1'], index=False)
