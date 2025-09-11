#Require libraries.
import numpy as np 
import pandas as pd 
import sklearn as svm
from  sklearn.metrics import accuracy_score
from  sklearn.preprocessing import StandardScaler
from  sklearn.model_selection import train_test_split
from  sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from  sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt 
import seaborn as sns
#Importing Dataset
dataset = pd.read_csv("Hypothyroidism_data.csv")

#Datafram Object as pandas
print(type(dataset))

#Shape of dataset
print(dataset.shape)

#Columns in Dataset
print(dataset.head(5))
print(dataset.sample(5))

#Description
print(dataset.describe())
print(dataset.info())

#Columns Understanding of dataset
info=["Age","Sex:1 = Male;0 = Female","On Thyroxine (1 = Yes; 0 = No)","TSH (mIU/L)","T3 measured(pmol/L):1 = Yes; 0 = No","T3  (pmol/L)","TT4 (μg/dL)","BinaryClass:0= NO_Hypothyroidism Disease,1=Hypothyroidism Disease"]
for i in range (len(info)):
  print(dataset.columns[i]+"\t\t", info[i])
  print(dataset['BinaryClass'].unique())

#Check Correlation between dataset
print(dataset.corr()['BinaryClass'].abs().sort_values(ascending=False))

#EDA(Exploratory Data Analysis)
y=dataset['BinaryClass']
sns.catplot(y)
target_temp= dataset.BinaryClass.value_counts()

#Analysis Features on dataset
#Age and BinaryClass
print(dataset['Age'].unique())
print(sns.catplot(x=dataset['Age'],y=dataset['BinaryClass']))
plt.show()

#SEX (FEMALE,MALE)and BinaryClass
print(dataset['Sex'].unique())
print(sns.catplot(x=dataset["Sex"],y=dataset["BinaryClass"]))
plt.show()

#On Thyroxine(Y/N) and BinaryClass
print(dataset['On thyroxine'].describe())
print(dataset['On thyroxine'].unique())
print(sns.catplot(data=dataset,x=dataset['On thyroxine'],y=dataset['BinaryClass']))
plt.show()

#TSH  Analysis (Numerical Data) and BinaryClass
print(dataset['TSH'].describe())
print(dataset['TSH'].unique())
print(sns.catplot(data=dataset,x=dataset['TSH'],y=dataset['BinaryClass']))
plt.show()

#T3 Mesured (Y/N) and BinaryClass
print(dataset['T3 measured'].describe())
print(dataset['T3 measured'].unique())
print(sns.catplot(data=dataset,x=dataset['T3 measured'],y=dataset['BinaryClass']))
plt.show()

#T3 Analysis (Numerical Data) and BinaryClass
print(dataset['T3'].describe())
print(dataset['T3'].unique())
print(sns.catplot(data=dataset,x=dataset['T3'],y=dataset['BinaryClass']))
plt.show()

#T4 Analysis(Numerical Data) and BinaryClass
print(dataset['TT4'].describe())
print(dataset['TT4'].unique())
print(sns.catplot(data=dataset,x=dataset['TT4'],y=dataset['BinaryClass']))
plt.show()

#//////////////////////////////////////////
#TRAIN TEST SPLIT
Features=dataset.drop(columns=['Age','BinaryClass'],axis=1)
# 'features' now contains all columns from 'dataset' EXCEPT 'Age' and 'BinaryClass'
Target=dataset['BinaryClass']
#SPLIT THE DATA
x_train,x_test,y_train,y_test=train_test_split(Features,Target,test_size=0.2,random_state=41)
print(x_train.shape)
print(x_test.shape)
print(y_test.shape)
print(y_train.shape)

#Now Train the model
#Predict the accuracy,Precision,F1_Score,Classification Report,Recall..

#Train model With Support Vector Machine(SVM)
SVM_model = SVC(kernel='linear')
print(SVM_model.fit(x_train,y_train))

#Train Data
SVM_x_train_prediction = SVM_model.predict(x_train)
SVM_Traindata_accuracy = accuracy_score(y_train,SVM_x_train_prediction)
print("The accuracy score achieved using Train data of SVM:", SVM_Traindata_accuracy)
Precision_train_SVM=precision_score(y_train,SVM_x_train_prediction)
print("The precision score achieved using Train data of SVM:",Precision_train_SVM)
Recall_train_SVM= recall_score(y_train,SVM_x_train_prediction)
print("The recall score achieved using Train data of SVM :",Recall_train_SVM)
f1_train_SVM = f1_score(y_train,SVM_x_train_prediction)
print("The F1 score achieved using Train data of SVM:",f1_train_SVM)
Classification_Report_SVM_Train_Data=classification_report(y_train,SVM_x_train_prediction)
print("The classification score achieved using Train data of SVM:",Classification_Report_SVM_Train_Data)

#TEST DATA
SVM_x_test_prediction=SVM_model.predict(x_test)
SVM_data_accuracy= accuracy_score(y_test,SVM_x_test_prediction)
print("The accuracy score score achieved using Test data of SVM :",SVM_data_accuracy)
Precision_test_SVM=precision_score(y_test,SVM_x_test_prediction)
print("The precision score score achieved using Test data of SVM:",Precision_test_SVM)
Recall_test_SVM= recall_score(y_test,SVM_x_test_prediction)
print("The recall score score achieved using Test data of SVM:",Recall_test_SVM)
f1_test_SVM = f1_score(y_test,SVM_x_test_prediction)
print("The F1 score score achieved using Test data of SVM:",f1_test_SVM)
Classification_Report_SVM_Test_Data = classification_report(y_test, SVM_x_test_prediction)
print("The classification score score achieved using Test data of SVM:",Classification_Report_SVM_Test_Data)


#////////////////////////////
#Random Forest Train Model
RF_model =RandomForestClassifier(n_estimators=90, max_depth=3,random_state=41)
RF_model.fit(x_train, y_train)
#Train data
RF_x_train_prediction = RF_model.predict(x_train)
RF_train_data_accuracy = accuracy_score(y_train, RF_x_train_prediction)
print("The accuracy score achieved using Train data of Random Forest",RF_train_data_accuracy)
Precision_train_RF= precision_score(y_train,RF_x_train_prediction)
print("The precision score achieved using Train data of Random Forest:",Precision_train_RF)
Recall_train_RF= recall_score(y_train,RF_x_train_prediction)
print("The recall score achieved using Train data of Random Forest :",Recall_train_RF)
f1_train_RF = f1_score(y_train,RF_x_train_prediction)
print("The F1 score achieved using Train data of Random Forest :",f1_train_RF)
Classification_Report_RF_Train_Data=classification_report(y_train,RF_x_train_prediction)
print("The classification score achieved using Train data of Random Forest:",Classification_Report_RF_Train_Data)

# Test Data
RF_x_test_prediction= RF_model.predict(x_test)  
RF_test_data_accuracy = accuracy_score(y_test,RF_x_test_prediction)
print("The accuracy score achieved using Test data of Random Forest:",RF_test_data_accuracy)
Precision_test_RF=precision_score(y_test,RF_x_test_prediction)
print("The precision score achieved using Test data of Random Forest:",Precision_test_RF)
Recall_test_RF= recall_score(y_test,RF_x_test_prediction)
print("The recall score achieved using Test data of Random Forest :",Recall_test_RF)
f1_test_RF = f1_score(y_test,RF_x_test_prediction)
print("The F1 score achieved using Test data of Random Forest:",f1_test_RF)
Classification_Report_RF_Test_Data = classification_report(y_test,RF_x_test_prediction)
print("The classification score achieved using Test data of Random Forestt:",Classification_Report_RF_Test_Data)


#//////////////////////////////
#Logistic Regression
lr_model=LogisticRegression()
lr_model.fit(x_train,y_train)
#Train Data
lr_x_train_prediction=lr_model.predict(x_train)
lr_train_data_accuracy=accuracy_score(y_train,lr_x_train_prediction)
print("The accuracy score achieved using Train data of Logistic Regression is:",lr_train_data_accuracy)
Precision_train_lr= precision_score(y_train,lr_x_train_prediction)
print("The precision score achieved using Train data of Logistic Regression is:",Precision_train_lr)
Recall_train_lr= recall_score(y_train,lr_x_train_prediction)
print("The recall score achieved using Train data of Logistic Regression is :",Recall_train_lr)
f1_train_lr = f1_score(y_train,lr_x_train_prediction)
print("The F1 score achieved using Train data of Logistic Regression is :",f1_train_lr)
Classification_Report_lr_Train_Data=classification_report(y_train,lr_x_train_prediction)
print("The classification score achieved using Train data of Logistic Regression is:",Classification_Report_lr_Train_Data)
#Test Data
lr_x_test_prediction=lr_model.predict(x_test)
lr_test_data_accuracy=accuracy_score(y_test,lr_x_test_prediction)
print("The accuracy score achieved using Test data of Logistic Regression is :",lr_test_data_accuracy)
Precision_test_lr=precision_score(y_test,lr_x_test_prediction)
print("The precision score achieved using Test data of Logistic Regression is :",Precision_test_lr)
Recall_test_lr= recall_score(y_test,lr_x_test_prediction)
print("The recall score achieved using Test data of Logistic Regression is :",Recall_test_lr)
f1_test_lr = f1_score(y_test,lr_x_test_prediction)
print("The F1 score achieved using Test data of Logistic Regression is:",f1_test_lr)
Classification_Report_lr_Test_Data = classification_report(y_test,lr_x_test_prediction)
print("The classification score achieved using Test data of Logistic Regression is:",Classification_Report_lr_Test_Data)

#////////////////////////////

#PREDICTION SYSTEM1
Inputdata = ("Enter your : Sex, On thyroxine, TSH, T3 measured, T3, TT4")
Inputdata_numpy    = np.asarray(Inputdata)
Inputdata_reshaped = Inputdata_numpy.reshape(1,-1)
Prediction1        = SVM_model.predict(Inputdata_reshaped)
Prediction2        = RF_model.predict(Inputdata_reshaped)
Prediction3        = lr_model.predict(Inputdata_reshaped)
PREDICTION = [Prediction1,Prediction2,Prediction3]
print(PREDICTION)
if(PREDICTION[0]==0):
     print("The patient shows no signs of Hypothyroidism.")
else:
    print("The patient shows signs of Hypothyroidism.")    

#////////////////////////////