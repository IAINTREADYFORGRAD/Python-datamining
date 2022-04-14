# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 01:07:51 2021

@author: user
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold,cross_val_score
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib

data = pd.read_csv('diabetes.csv')
data.shape  #查看此筆資料的行列，是多少成多少的矩陣
data.info() #查看此筆資料有哪些資料

#妊娠糖尿病是因孕婦身體對碳水化合物的耐受度降低
 ，出現血糖過高的症狀，孕程中的人體代謝會隨著各種荷爾蒙分泌而出現變化
 ，導致人體增加胰島素的需求，進而產生妊娠糖尿病
#大部分的妊娠糖尿病會在生產完後恢復正常
#若飲食及生活習慣沒有控管好，有70％～80％的機率得到糖尿病
data.Outcome.value_counts() #查看此筆資料有多少個人患有糖尿病

outcome=['healthy','diabetic']
bar=plt.bar(outcome,data.Outcome.value_counts(), #設定bar的x, y標示
        color=['lightsteelblue','cornflowerblue'])
plt.title('Count of Outcome')
for item in bar: #遍歷每個柱子
       height = item.get_height()
       plt.text(item.get_x()+item.get_width()/2.3, height *0.85, int(height))
plt.title('Count of Outcome')
plt.show()

data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
#將理論上不可以等於0的重要資料，用NaN(null)代替0
data.isnull().sum() #查看各個項目有沒有缺失值
def mean_target(var):   
    temp1 = data[data[var].notnull()]
    temp1 = temp1[[var, 'Outcome']].groupby(['Outcome'])[[var]].mean().reset_index()
    return temp1
def median_target(var):   
    temp2 = data[data[var].notnull()]
    temp2 = temp2[[var, 'Outcome']].groupby(['Outcome'])[[var]].median().reset_index()
    return temp2
#決定要用中間值或是用平均值填補缺失值
mean_target('Glucose')
median_target('Glucose')
#中間值與平均值，兩種數值都差不多
#正常餐後兩小時(此筆資料為餐露兩小時測的)血糖值應小於140 mg/dL 
#糖尿病患者餐後兩小時血糖值應大於140 mg/dL
mean_target('Insulin')
median_target('Insulin')
#中間值與平均值差異大
#正常餐後兩小時胰島素值應介於9.93-124.9 pmol/ml
#糖尿病患者餐後兩小時胰島素值會小於9.93pmol/ml 或者 大於124.9 pmol/ml (查為甚麼大於)
#未患有糖尿病受試者平均值超出了正常值，中間值則在正常值裡
#因此整筆資料用中間值取代缺失值
median_target('Glucose')
data.loc[(data['Outcome'] == 0 ) & (data['Glucose'].isnull()), 'Glucose'] = 107
data.loc[(data['Outcome'] == 1 ) & (data['Glucose'].isnull()), 'Glucose'] = 140
median_target('Insulin')
data.loc[(data['Outcome'] == 0 ) & (data['Insulin'].isnull()), 'Insulin'] = 102.5
data.loc[(data['Outcome'] == 1 ) & (data['Insulin'].isnull()), 'Insulin'] = 169.5
median_target('SkinThickness')
data.loc[(data['Outcome'] == 0 ) & (data['SkinThickness'].isnull()), 'SkinThickness'] = 27
data.loc[(data['Outcome'] == 1 ) & (data['SkinThickness'].isnull()), 'SkinThickness'] = 32
median_target('BloodPressure')
data.loc[(data['Outcome'] == 0 ) & (data['BloodPressure'].isnull()), 'BloodPressure'] = 70
data.loc[(data['Outcome'] == 1 ) & (data['BloodPressure'].isnull()), 'BloodPressure'] = 74.5
median_target('BMI')
data.loc[(data['Outcome'] == 0 ) & (data['BMI'].isnull()), 'BMI'] = 30.1
data.loc[(data['Outcome'] == 1 ) & (data['BMI'].isnull()), 'BMI'] = 34.3
data.isnull().sum() #確認缺失值都有被補上


X = data.loc[:,'Pregnancies':'Age']
y = data.loc[:,['Outcome']]
X_train,X_test,y_train,y_test=train_test_split(
X.values,y.values,random_state=0,test_size=0.25)

knn1=KNeighborsClassifier(n_neighbors=3)
knn1.fit(X_train,y_train)
knn1.score(X_train,y_train)
knn1.score(X_test,y_test) #擬合度滿低的


knn2_score=[]
best_prediction2=[0,0] # 第一個0 (index=0) 等等要放i，第二個0 (index=1)等等要放score
for i in range(1,100): #k是多找時準確率最高
    knn2 = KNeighborsClassifier(n_neighbors=i)
    knn2.fit(X_train,y_train)
    score = knn2.score(X_test,y_test)
    knn2_score.append(score)
    if score > best_prediction2[1]: #如果score大於best_prediction index 1的數值
        best_prediction2=[i,score] 
print(best_prediction2) 
plt.plot(range(1,100),knn2_score)
#[13, 0.9166666666666666]

knn13=KNeighborsClassifier(n_neighbors=13).fit(X_train,y_train)
knn13.score(X_train,y_train) #0.8715277777777778
knn13.score(X_test,y_test) # 0.9166666666666666
#因為訓練集和測試集是隨機分配的，測試結果具有隨機性，不能用於判斷演算法好壞。
 所以，多次隨機分配訓練集和交叉驗證測試集，然後對結果取平均值再比較：
score_knn13 = cross_val_score(knn13,X,y,cv=10,scoring='accuracy')
print(score_knn13.mean()) 
#交叉驗證，當k=10時分成10槽做交叉驗證的平均準確率

x_scatter = data.Glucose
y_scatter = data.BMI
plt.xlabel('Glucose')
plt.ylabel('BMI')

#看看決策樹訓練出來的結果
tree_clf =  tree.DecisionTreeClassifier().fit(X_train, y_train)
tree_clf.score(X_train,y_train) #1.0
tree_clf.score(X_test,y_test) #0.85937
score_tree= cross_val_score(tree_clf,X,y,cv=10,scoring='accuracy')
print(score_tree.mean()) #0.8398667122351332

#看看隨機森林訓練出來的結果 
forest_clf=RandomForestClassifier(n_estimators=10) 
forest=forest_clf.fit(X_train, y_train) #森林樹木10個
forest.score(X_train,y_train) #1.0
forest.score(X_test,y_test) #0.90625
score_forest = cross_val_score(forest,X,y,cv=10,scoring='accuracy')
print(score_forest.mean()) #0.8783928571428571

#看看邏輯斯回歸訓練出來的結果 
Logistic=LogisticRegression().fit(X_train, y_train)
Logistic.score(X_train,y_train) #0.77951
Logistic.score(X_test,y_test) #0.78125
score_Logistic = cross_val_score(Logistic,X,y,cv=10,scoring='accuracy')
print(score_Logistic.mean()) #0.7773410799726589

#把knn、決策樹、隨機森林、邏輯斯回歸的訓練結果放一起
ML = pd.DataFrame({"Models":["KNN","Decision Tree",
                                "Random Forest","Logistic Regression"],
                     "Score":[knn13.score(X_test,y_test),
                              tree_clf.score(X_test,y_test),
                              forest.score(X_test,y_test),
                              Logistic.score(X_test,y_test)]})
print(ML)



#為什麼邏輯斯回歸的準確率會好低
#knn混在一起

#看看預測判斷是否患有糖尿病重要的特徵
diabetes_features = [x for i, x in enumerate(data.columns) if i!=8]
def plot_feature_importances_diabetes(model):
    n_features = 8
    plt.barh(range(n_features), model.feature_importances_,color=['lightsteelblue'])
    plt.yticks(np.arange(n_features), diabetes_features)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
plot_feature_importances_diabetes(forest)
plt.savefig('feature_importance')

joblib.dump(forest_clf,'diabetes_forest_clf.kpl')
!dir diabetes_forest_clf.*
loadforest=joblib.load('diabetes_forest_clf.kpl')

#使用者介面
Age=int(input('年齡:'))
DiabetesPredigreeFunction=float(input('糖尿病家族函數:'))
BMI=float(input('BMI:'))
Insulin=float(input('飯後兩小時血清胰島素:'))
Glucose=float(input('後兩小時血糖:'))
SkinThickness=float(input('皮褶厚度(全身脂肪含量):'))
BloodPressure=float(input('血壓:'))
Pregnancies=int(input('生過幾胎:'))

input_list=[Age, DiabetesPredigreeFunction, 
            BMI, Insulin, Glucose, SkinThickness, 
            BloodPressure, Pregnancies]
input_array=np.array(input_list).reshape(1,-1)
pred=loadforest.predict(input_array)

dict_diabetes={0:'NO', 1:'YES' }
print('年齡: ', Age,"\n",
      '糖尿病家族函數: ', DiabetesPredigreeFunction,"\n",
      'BMI: ', BMI,'\n',
      '飯後兩小時血清胰島素:', Insulin,'\n',
      '後兩小時血糖: ', Glucose,'\n',
      '皮褶厚度(全身脂肪含量): ', SkinThickness,'\n',
      '血壓: ', BloodPressure,'\n'
      '生過幾胎: ', Pregnancies,'\n')
print("是否患有糖尿病: ", dict_diabetes[pred[0]])


      
      






























