# Loading the required libraries #

from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import (
    GridSearchCV, KFold, train_test_split, cross_val_score)
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import numpy as np
import pandas as pd

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Avoid Warnings
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

# Common model helpers


# Read in water_potability file #

waterDf = pd.read_csv('../water_potability.csv')

# Make a copy #

waterData = waterDf.copy()

# About the data #

print('The water-potability file has')
print('   Rows      Columns')
print('   {}         {}\n' .format(waterData.shape[0], waterData.shape[1]))

print(waterData.info())

print('Information about features\n')
print(waterData.describe())

# How does the data look like? #
print('How does the water-potability data look like?\n')
print(waterData.head())

# We work on the missing data #
print('There are missing values within the data.\n')
print('The nature of the missing values within the features are as follows:\n')
print(waterData.isna().sum())

# Imputing 'ph' value #

phMean_0 = waterData[waterData['Potability'] == 0]['ph'].mean(skipna=True)
waterData.loc[(waterData['Potability'] == 0) & (
    waterData['ph'].isna()), 'ph'] = phMean_0
phMean_1 = waterData[waterData['Potability'] == 1]['ph'].mean(skipna=True)
waterData.loc[(waterData['Potability'] == 1) & (
    waterData['ph'].isna()), 'ph'] = phMean_1

# Imputing 'Sulfate' value #

SulfateMean_0 = waterData[waterData['Potability']
                          == 0]['Sulfate'].mean(skipna=True)
waterData.loc[(waterData['Potability'] == 0) & (
    waterData['Sulfate'].isna()), 'Sulfate'] = SulfateMean_0
SulfateMean_1 = waterData[waterData['Potability']
                          == 1]['Sulfate'].mean(skipna=True)
waterData.loc[(waterData['Potability'] == 1) & (
    waterData['Sulfate'].isna()), 'Sulfate'] = SulfateMean_1

# Imputing 'Trihalomethanes' value #

TrihalomethanesMean_0 = waterData[waterData['Potability']
                                  == 0]['Trihalomethanes'].mean(skipna=True)
waterData.loc[(waterData['Potability'] == 0) & (
    waterData['Trihalomethanes'].isna()), 'Trihalomethanes'] = TrihalomethanesMean_0
TrihalomethanesMean_1 = waterData[waterData['Potability']
                                  == 1]['Trihalomethanes'].mean(skipna=True)
waterData.loc[(waterData['Potability'] == 1) & (
    waterData['Trihalomethanes'].isna()), 'Trihalomethanes'] = TrihalomethanesMean_1

# Check #
print('Checking to see any more missing data \n')
waterData.isna().sum()

# Convert 'Potability' to Category #

waterData['Potability'] = waterData['Potability'].astype('category')
waterData.info()

print('Distribution of Target Variable within the sample data')

fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(16, 6))

waterData.Potability.value_counts().plot(
    kind='bar', color=['orange', 'steelblue'], rot=0, ax=ax[0])
# Iterrating over the bars one-by-one
for bar in ax[0].patches:
    ax[0].annotate(format(bar.get_height(), 'd'), (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                   ha='center', va='center', size=15, xytext=(0, -10),
                   textcoords='offset points')
ax[0].tick_params(left=False, labelleft=False)
ax[0].xaxis.set_tick_params(labelsize=20)

labels = list(waterData['Potability'].unique())
# use the wedgeprops and textprops arguments to style the wedges and texts, respectively
ax[1].pie(waterData['Potability'].value_counts(), labels=labels, autopct='%1.1f%%',
          colors=['orange', 'steelblue'], explode=[0.005]*len(labels),
          textprops={'size': 'x-large'},
          wedgeprops={'linewidth': 3.0, 'edgecolor': 'white'})

plt.show()

# Correlation Matrix #

Corrmat = waterData.corr()
plt.subplots(figsize=(7, 7))
sns.heatmap(Corrmat, cmap="YlGnBu", square=True, annot=True, fmt='.2f')
plt.show()

print('Boxplot and density distribution of different features by Potability\n')

fig, ax = plt.subplots(ncols=2, nrows=9, figsize=(14, 28))

features = list(waterData.columns.drop('Potability'))
i = 0
for cols in features:
    sns.kdeplot(waterData[cols], fill=True, alpha=0.4, hue=waterData.Potability,
                palette=('indianred', 'steelblue'), multiple='stack', ax=ax[i, 0])

    sns.boxplot(data=waterData, y=cols, x='Potability', ax=ax[i, 1],
                palette=('indianred', 'steelblue'))
    ax[i, 0].set_xlabel(' ')
    ax[i, 1].set_xlabel(' ')
    ax[i, 1].set_ylabel(' ')
    ax[i, 1].xaxis.set_tick_params(labelsize=14)
    ax[i, 0].tick_params(left=False, labelleft=False)
    ax[i, 0].set_ylabel(cols, fontsize=16)
    i = i+1

plt.show()

print('Correlation of Potability with feature variables:')
features = list(waterData.columns.drop('Potability'))

Corr = list()
for cols in features:
    Corr.append(waterData[cols].corr(waterData['Potability']))

corrDf = pd.DataFrame({'Features': features, 'Corr': Corr})
corrDf['Corr'] = corrDf['Corr'].abs()
corrDf.sort_values(by='Corr', ascending=True)

# Preparing the Data for Modelling #

X = waterData.drop('Potability', axis=1).copy()
y = waterData['Potability'].copy()

# Train-Test split #
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Synthetic OverSampling #
print('Balancing the data by SMOTE - Oversampling of Minority level\n')
smt = SMOTE()
counter = Counter(y_train)
print('Before SMOTE', counter)
X_train, y_train = smt.fit_resample(X_train, y_train)
counter = Counter(y_train)
print('\nAfter SMOTE', counter)

# Scaling #
ssc = StandardScaler()

X_train = ssc.fit_transform(X_train)
X_test = ssc.transform(X_test)

modelAccuracy = list()

model = [LogisticRegression(), DecisionTreeClassifier(), GaussianNB(), RandomForestClassifier(), ExtraTreesClassifier(),
         svm.LinearSVC(), XGBClassifier(), CatBoostClassifier()]
trainAccuracy = list()
testAccuracy = list()
kfold = KFold(n_splits=10, random_state=7, shuffle=True)

for mdl in model:
    trainResult = cross_val_score(
        mdl, X_train, y_train, scoring='accuracy', cv=kfold)
    trainAccuracy.append(trainResult.mean())
    mdl.fit(X_train, y_train)
    y_pred = mdl.predict(X_test)
    testResult = metrics.accuracy_score(y_test, y_pred)
    testAccuracy.append(testResult)

print('The comparision\n')
modelScore = pd.DataFrame(
    {'Model': model, 'Train_Accuracy': trainAccuracy, 'Test_Accuracy': testAccuracy})
modelScore

# RandomForestClassfier #
print('Random Forest Classifier\n')
Rfc = RandomForestClassifier()
Rfc.fit(X_train, y_train)

y_Rfc = Rfc.predict(X_test)
print(metrics.classification_report(y_test, y_Rfc))
print(modelAccuracy.append(metrics.accuracy_score(y_test, y_Rfc)))

sns.heatmap(confusion_matrix(y_test, y_Rfc), annot=True, fmt='d')
plt.show()

# XGB Classifier() #
print('XGB Classifier\n')
xgb = XGBClassifier()
xgb.fit(X_train, y_train)

y_xgb = xgb.predict(X_test)
print(metrics.classification_report(y_test, y_xgb))
print(modelAccuracy.append(metrics.accuracy_score(y_test, y_xgb)))

sns.heatmap(confusion_matrix(y_test, y_xgb), annot=True, fmt='d')
plt.show()

# CatBoostClassifier() #
print('CatBoostClassifier\n')
cat = CatBoostClassifier(verbose=False)
cat.fit(X_train, y_train)

y_cat = cat.predict(X_test)
print(metrics.classification_report(y_test, y_cat))
print(modelAccuracy.append(metrics.accuracy_score(y_test, y_cat)))

sns.heatmap(confusion_matrix(y_test, y_cat), annot=True, fmt='d')
plt.show()
