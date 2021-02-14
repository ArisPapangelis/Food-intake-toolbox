# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 17:38:11 2021

@author: Aris Papangelis

Script to train a supervised learning classifier

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn import preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, KFold, LeaveOneOut, LeavePOut, LeaveOneGroupOut
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.svm import SVC, LinearSVC 
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

#Clemson cafeteria dataset
dataset = pd.read_csv("csv/clemson_indicators.csv", delimiter = ';')
cols_to_use = ['TP', 'FP', 'FN', 'Precision','Recall', 'F1']   
dataset = pd.concat([dataset, pd.read_csv("csv/bite_metrics.csv", usecols=cols_to_use, delimiter = ';')], axis=1)
dataset['BMI'] = [0.0]* len(dataset)

cols_to_use = ["Participant","Gender","Age","BMI"]
demographics = pd.read_csv("csv/clemson_demographics.csv", usecols = cols_to_use, delimiter = ';')
for i in range(len(dataset)):
    name = dataset.at[i,'Participant'].split('_')[0]
    bmi = demographics[demographics['Participant']==name]['BMI']
    dataset.at[i,'BMI'] = bmi

dataset = dataset[dataset['F1'] > 0.6]


#Dataset katerinas
dataset2 = pd.read_csv("csv/katerina_indicators.csv", delimiter = ';')
dataset2 = pd.concat([dataset2, pd.read_csv("csv/katerina_demographics.csv", usecols=['BMI'], delimiter = ';')], axis=1)

cols_to_use = ['Participant', 'a', 'b', 'Total food intake','Average food intake rate', 
        'Average bite size', 'Bites per minute', 'BMI']


#Final data
data = pd.concat([dataset[cols_to_use], dataset2[cols_to_use]], ignore_index=True)
#data = dataset[cols_to_use]

X = data.iloc[:,1:-1]
Y = data.iloc[:,-1]

#Rescale data to range (0,1)
data_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
#data_scaler = preprocessing.RobustScaler()
X_rescaled = data_scaler.fit_transform(X)

#0 for normal, 1 for overweight or underweight
Y = np.where((Y < 18.5) | (Y > 25), 1, 0)


#Data visualisation
df = data.iloc[:,1:]
df['BMI'] = Y
df.iloc[:,:-1] = X_rescaled

cor = df.corr()
plt.figure(figsize=(19.2, 10.8))
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)

plt.figure(figsize=(19.2, 10.8))
plt.suptitle('Features')
i=1
for col in df.columns[:-1]:
    plt.subplot(3, 2, i)
    #plt.title(col)
    row = sns.boxplot( x='BMI', y=col, data= df)
    i+=1
    
"""
#Select best features through SelectKBest
features = SelectKBest(score_func=f_classif, k=3)
fit = features.fit(X_rescaled, Y)
print(fit.scores_)
X_rescaled = fit.transform(X_rescaled)
"""


#Select best features through RFE
model = ExtraTreesClassifier(random_state=0)
rfe = RFE(model, 3)
fit = rfe.fit(X_rescaled, Y)
print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)
X_rescaled = X_rescaled[:,fit.support_]


"""
#Do PCA on X_rescaled
pca = PCA()
fit = pca.fit(X_rescaled)

print("Explained Variance: %s" % fit.explained_variance_ratio_)
print(fit.components_)
"""


#Classification
x_train, x_test, y_train, y_test = train_test_split(X_rescaled, Y, test_size=0.2, random_state=0)
#model = RandomForestClassifier(random_state=0).fit(x_train, y_train)
model = ExtraTreesClassifier(random_state=0).fit(x_train, y_train)
#model = KNeighborsClassifier(n_neighbors=3).fit(x_train, y_train)
#model = DecisionTreeClassifier(random_state=0).fit(x_train, y_train)
#model = GaussianNB().fit(x_train, y_train)
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test,y_pred)
cm = confusion_matrix(y_test,y_pred)
ROC = roc_auc_score(y_test, y_pred)


#Check Algorithms
#Decision tree has high recall and f1, random forest has balance between recall and accuracy, as well as best roc_auc
#If we prefer FP to FN (normal classified as overweight) decision tree is better
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier(n_neighbors=3)))
models.append(('CART', DecisionTreeClassifier()))
models.append(('RF', RandomForestClassifier()))
models.append(('ET', ExtraTreesClassifier()))
models.append(('ADA', AdaBoostClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
    #kfold = LeavePOut(100)  
    cv_results = cross_val_score(model, X_rescaled, Y, cv=kfold, scoring='roc_auc')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

