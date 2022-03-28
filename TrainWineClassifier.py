import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Ease data preprocessing
from sklearn import preprocessing

# Import models
from sklearn.ensemble import RandomForestClassifier

# PCA
from sklearn.decomposition import PCA

# Import cross-validation tools
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

# Import metrics to evaluate model performance
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Use model in future
import joblib

# downloading dataset into pandas dataframe
# url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv('winequality-red.csv', sep=';')

# checking data - all integers
data.head()

# # data needs standardising
# data.describe()
# # managable size
# data.shape()
# # different quality count
# data.groupby('quality').size()
# # no NAs
# data.isna().values.any()


# three classes - bad, average and good
targets = []

for q in data['quality']:
    if q <= 4:
        targets.append(1)
    elif 5 <= q <= 6:
        targets.append(2)
    elif q >= 7:
        targets.append(3)
        
data['target'] = targets

# # skewed data
# data.groupby('target').size()

# no need of quality now
data = data.drop('quality', axis=1)

# # no strong correlation
# data.corr()*100

# test and train split
y = data.target
X = data.drop(['target'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=123)

# pipelining
pipeline = make_pipeline(preprocessing.StandardScaler(),RandomForestClassifier(n_estimators=100))

# Removing useless features
pca = PCA()  
X_train = pca.fit_transform(X_train)  
X_test = pca.transform(X_test)

# the variance caused by each feature on the dataset
explained_variance = pca.explained_variance_ratio_ 
for i in explained_variance:
    print(format(i*100, 'f'))

# the four first features of our data capture almost 99.5% of the variance
pca = PCA(n_components=4)  
X_train = pca.fit_transform(X_train)  
X_test = pca.transform(X_test)

# hyper parameters for our model
print(pipeline.get_params())

# the hyper parameters we want to tune through cross-validation
hyperparameters = { 'randomforestclassifier__max_features' : ['auto', 'sqrt', 'log2'],'randomforestclassifier__max_depth' : [None, 10, 7, 5, 3, 1]}

# performs cross-validation across all possible permutations of hyper parameters
clf = GridSearchCV(pipeline, hyperparameters, cv=10)
clf.fit(X_train, y_train)

# best parameters
print(clf.best_params_)

# np.savetxt('test.out', X_test, delimiter=',')

# # predicting over test set
# y_pred = clf.predict(X_test)

# # confusion matrix to check how the model classified the different wines on the dataset
# print('Accuracy score:', accuracy_score(y_test, y_pred))
# print("-"*80)
# print('Confusion matrix\n')
# conmat = np.array(confusion_matrix(y_test, y_pred, labels=[1,2,3]))
# confusion = pd.DataFrame(conmat, index=['Actual 1', 'Actual 2', 'Actual 3'],
#                          columns=['predicted 1','predicted 2', 'predicted 3'])
# print(confusion)
# print("-"*80)
# print('Classification report')
# print(classification_report(y_test, y_pred, target_names=['1','2', '3']))

# save our previous model to apply it to future data
joblib.dump(clf, 'wine-classifier-model.pkl')
