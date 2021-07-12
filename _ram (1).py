#!/usr/bin/env python
# coding: utf-8



#sex- Gender of patient Male=1, Female =0
#Age-Age of patients
#Diabities-0= No, 1=Yes
#Anaemia-0 = No , 1=Yes
#High Blood pressure-0 = No, 1=Yes
#Smoking-0 =No, 1=Yes
#DEATH EVENT-0=No, 1=Yes





import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import plotly.graph_objs as go
import plotly.express as px





heart_data=pd.read_csv("heart_failure_clinical_records_dataset.csv")





heart_data.head()





heart_data.describe()





heart_data.isnull().sum()


# #pie charts



labels =['No diabetes','diabetes']
diabetes_yes = heart_data[heart_data['diabetes']==1]
diabetes_no =heart_data[heart_data['diabetes']==0]
values = [len(diabetes_no), len(diabetes_yes)]
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.5)])
fig.update_layout(
    title_text='Analysis on Diabetes')
fig.show()
         





fig=px.pie(heart_data,values='diabetes',names='DEATH_EVENT',title='Death Analysis')
fig.show()


# #heat map




plt.figure(figsize=(20,20))
sb.heatmap(heart_data.corr(),vmin=-1,cmap='coolwarm',annot=True);


# #data modeling




from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score





Feature=['time','ejection_fraction','serum_creatinine']
x=heart_data[Feature]
y=heart_data['DEATH_EVENT']





xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=2)





from sklearn.linear_model import LogisticRegression





log_re=LogisticRegression()





log_re.fit(xtrain,ytrain)
log_re_pred=log_re.predict(xtest)




log_acc=accuracy_score(ytest,log_re_pred)
print("Logistic Accuracy Score:","{:.2f}%)".format(100*log_acc))





from mlxtend.plotting import plot_confusion_matrix





cm = confusion_matrix(ytest, log_re_pred)
plt.figure()
plot_confusion_matrix(cm, figsize=(15,10), hide_ticks=True,cmap=plt.cm.Blues)
plt.title("Logistic Regression , Confusion Matrix")
plt.xticks(range(2),["Heart Not Failed", "Heart Failed"],fontsize=20)
plt.yticks(range(2),["Heart Not Failed","Heart Failed"],fontsize=20)
plt.show()







