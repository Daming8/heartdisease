#!/usr/bin/env python3
# Random Forest modelling
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score,recall_score
from sklearn.metrics import f1_score
df = pd.read_csv("/home/ubuntu/data/heart_scaled.csv")
y=df.HeartDiseaseYes
train=df.drop(["HeartDiseaseYes","HeartDiseaseNo"],axis=1)
X_train,X_test,y_train,y_test=train_test_split(train,
                                               y,
                                               test_size=0.3,
                                               random_state=42)
m=RandomForestClassifier()
m.fit(X_train,y_train)
y_pred=m.predict(X_test)
print(f'Model: {str(m)}')
a=accuracy_score(y_test, y_pred)
p=precision_score(y_test, y_pred)
r=recall_score(y_test, y_pred)
f=f1_score(y_test, y_pred)
print(f'Accuracy Score: {a}')
print(f'Precission Score: {p}')
print(f'Recall Score: {r}')
print(f'F1-Score: {f}')
print('-' * 100, '\n')
result= {'Model':['RF'],
        'Accuracy':[a],
        'Precision':[p],
        'Recall':[r],
        'F1':[f]}
result=pd.DataFrame(result)
result.to_csv('/home/ubuntu/data/result.csv', mode='a', index=False, header=False)
