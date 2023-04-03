import pandas as pd
cbb = pd.read_csv("cbb.csv")
cbb.head()

import seaborn as sns
import matplotlib.pyplot as plt

correlation = cbb.corr()
plt.subplots(figsize=(15,15))
sns.heatmap(correlation, annot=True)

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import f_oneway 
from statsmodels.stats.multicomp import pairwise_tukeyhsd

%matplotlib inline
pd.set_option("display.max_rows", 100, "display.max_columns", 100)

cbb = pd.read_csv("cbb.csv")


#Selecting teams that particapted in March Madness
Qaulified_For_MM = cbb[cbb['POSTSEASON'].notna()].reset_index(drop=True)
Postseason_labels = ['R68', 'R64','R32','S16','E8', 'F4','2ND','Champions']

#Creating a function to check if the p value of a set of data is greater than or equal to .05 and performing a Turkey test if true.
def F_Oneway_Turkey(stat):
    fstat, pval = f_oneway(*(Qaulified_For_MM[stat][Qaulified_For_MM['POSTSEASON'] == i] for i in Postseason_labels))
    if pval <= .05:
        return print(pairwise_tukeyhsd(Qaulified_For_MM[stat],Qaulified_For_MM['POSTSEASON'] ,.05))
    else:
        print('Not Stat Sig for ' +stat)





fig = plt.figure(figsize = (10,4))
sns.boxplot(data=Qaulified_For_MM, x='POSTSEASON', y='ADJOE', order=Postseason_labels)
plt.title('Offensive Adjusted Efficiency')

fig = plt.figure(figsize = (10,4))
sns.boxplot(data=Qaulified_For_MM, x='POSTSEASON', y='ADJDE', order=Postseason_labels)
plt.title('Defensive Adjusted Efficiency')

fig, ax = plt.subplots(1,2, figsize=(20,4))
sns.boxplot(ax= ax[0], data=Qaulified_For_MM, x='POSTSEASON', y='2P_O', order=Postseason_labels)
ax[0].set_title('Offensive 2 Point Shots Percentage Made')
sns.boxplot(ax= ax[1], data=Qaulified_For_MM, x='POSTSEASON', y='2P_D', order=Postseason_labels)
ax[1].set_title('Defensive 2 Point Shots Percentage Allowed')

fig, ax = plt.subplots(1,2, figsize=(20,4))
sns.boxplot(ax= ax[0], data=Qaulified_For_MM, x='POSTSEASON', y='3P_O', order=Postseason_labels)
ax[0].set_title('Offensive 3 Point Shot Percentage Made')
sns.boxplot(ax= ax[1], data=Qaulified_For_MM, x='POSTSEASON', y='3P_D', order=Postseason_labels)
ax[1].set_title('Defensive 3 Point Shot Percentage Allowed')

import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression

cbb = cbb.dropna()

pos_mapping = {'Champions':0,'2ND':1,'F4':2,'E8':3,'S16':4,'R32':5,'R64':6,'R68':7}
cbb['POSTSEASON'] = cbb['POSTSEASON'].map(pos_mapping)


conf_mapping = {'WCC':0,'ACC':1 ,'B10':2 ,'SEC':3, 'B12' :4,'Amer':5, 'BE' :6,'MAC':7, 'SC':8, 'MWC':9, 'A10':10, 'P12':11,
 'OVC': 12,'WAC':13, 'BSth':14, 'BW':15, 'AE': 16,'CAA':17, 'Ivy':18, 'Horz':19,'SB':20, 'CUSA':21, 'Pat':22, 'BSky':23,
 'MVC': 24,'Slnd':25, 'Sum' :26,'MAAC': 27,'SWAC':28, 'NEC':29 ,'MEAC':30, 'ASun':31}
cbb['CONF'] = cbb['CONF'].map(conf_mapping)

# toggle
# cbb['POSTSEASON'] = cbb['POSTSEASON'].fillna(8)
# cbb['CONF'] = cbb['CONF'].fillna(100)

# cbb['SEED'] = cbb['SEED'].fillna(30)

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import metrics 
from sklearn.metrics import confusion_matrix



X = cbb.drop(['POSTSEASON','TEAM'], axis=1)
y = cbb['POSTSEASON']

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.30, random_state = 5)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# cbb.head(10)

model1 = LogisticRegression(random_state=0, multi_class='multinomial', penalty='none', solver='newton-cg', max_iter=15000).fit(X_train, y_train)
preds = model1.predict(X_test)

import statsmodels.api as sm
import numpy as np

params = model1.get_params()
print(params)

# print('Intercept: \n', model1.intercept_)
# print('Coefficients: \n', model1.coef_)

np.exp(model1.coef_)

logit_model=sm.MNLogit(y_train,sm.add_constant(X_train))
logit_model
result=logit_model.fit()
stats1=result.summary()
stats2=result.summary2()
# print(stats1)
# print(stats2)

from sklearn import metrics
cm = metrics.confusion_matrix(y_test, preds)

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=model1.classes_)
disp.plot()

plt.show()

confusion_matrix(y_test, preds)
confmtrx = np.array(confusion_matrix(y_test, preds))
#pd.DataFrame(confmtrx, index=['0','1', '2','3','4','5','6','7'],
#columns=['predicted_1', 'predicted_2', 'predicted_3','predicted_4', 'predicted_5','predicted_6','predicted_7' ])

print('Accuracy Score:', metrics.accuracy_score(y_test, preds))

class_report=classification_report(y_test, preds)
print(class_report)

df = pd.read_csv('cbb.csv')
df = df.dropna()

# We are interested in POSTSEASON
X = df.loc[:, df.columns!='POSTSEASON']
# TEAM and CONF is not ideal to use
X = X.drop(['TEAM','CONF'], axis=1)
y = df['POSTSEASON']

from sklearn.preprocessing import MinMaxScaler
import sklearn.model_selection as ms
# Scale to make data balanced
scaler = MinMaxScaler().fit(X)
X_scaled = scaler.transform(X)
XTrain, XTest, YTrain, YTest = ms.train_test_split(X_scaled, y, test_size= 0.3, random_state=1)

import numpy as np
from sklearn import tree
from sklearn.model_selection import GridSearchCV

depth_val = np.arange(2,11)
leaf_val = np.arange(1,31, step=9)

grid_s = [{'max_depth': depth_val,'min_samples_leaf': leaf_val}]
model = tree.DecisionTreeClassifier(criterion='entropy')

cv_tree = GridSearchCV(estimator=model,param_grid=grid_s,cv=ms.KFold(n_splits=10), n_jobs=-1)
cv_tree.fit(XTrain, YTrain)

best_depth = cv_tree.best_params_['max_depth']

best_min_samples = cv_tree.best_params_['min_samples_leaf']

print(best_depth, best_min_samples)

model = tree.DecisionTreeClassifier(criterion='entropy',max_depth=best_depth,min_samples_leaf=best_min_samples)

cbbtree = model.fit(XTrain, YTrain)

y_pred = cbbtree.predict(XTest)

y_proba = cbbtree.predict_proba(XTest)

from sklearn.metrics import accuracy_score
print("accuracy:", accuracy_score(YTest, y_pred))

from sklearn import metrics
cm = metrics.confusion_matrix(YTest, y_pred)

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=cbbtree.classes_)
disp.plot()

plt.show()

class_report=classification_report(YTest, y_pred)
print(class_report)

tree.export_graphviz(cbbtree, out_file='cbbtree.dot', max_depth=best_depth, feature_names=X.columns)
