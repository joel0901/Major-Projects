import numpy as np
import pandas as pd
import extractor

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter("ignore", DeprecationWarning)

# data extraction

# the required features area extracted for all people into separate objects
# 'Class' assigns numeric value to the corresponing user which is the target value for this porblem

X_arav = extractor.features("data/aravind/")
X_arav = pd.DataFrame(X_arav)
X_arav['Class']=0
print('\nDataset 1 extracted')

X_joel = extractor.features("data/joel/")
X_joel = pd.DataFrame(X_joel)
X_joel['Class']=1
print('\nDataset 2 extracted')

X_christ = extractor.features("data/christ/")
X_christ = pd.DataFrame(X_christ)
X_christ['Class']=2
print('\nDataset 3 extracted')

X_shubham = extractor.features("data/shubham/")
X_shubham = pd.DataFrame(X_shubham)
X_shubham['Class']=3
print('\nDataset 4 extracted')

X_sourabh = extractor.features("data/sourabh/")
X_sourabh = pd.DataFrame(X_sourabh)
X_sourabh['Class']=4
print('\nDataset 5 extracted')

X_sourav = extractor.features("data/sourav/")
X_sourav = pd.DataFrame(X_sourav)
X_sourav['Class']=5
print('\nDataset 6 extracted')

X_sudip = extractor.features("data/sudip/")
X_sudip= pd.DataFrame(X_sudip)
X_sudip['Class']=6
print('\nDataset 7 extracted')

print('\nData Extraction Complete.')

# defining training and testing data
# by dividing in the ratio of 80-20% for train and test respectively

X_arav_train = X_arav[:int(X_arav.shape[0]*0.8)]
X_arav_test = X_arav[int(X_arav.shape[0]*0.8):]

X_joel_train = X_joel[:int(X_joel.shape[0]*0.8)]
X_joel_test = X_joel[int(X_joel.shape[0]*0.8):]

X_christ_train = X_christ[:int(X_christ.shape[0]*0.8)]
X_christ_test = X_christ[int(X_christ.shape[0]*0.8):]

X_shubham_train = X_shubham[:int(X_shubham.shape[0]*0.8)]
X_shubham_test = X_shubham[int(X_shubham.shape[0]*0.8):]

X_sourabh_train = X_sourabh[:int(X_sourabh.shape[0]*0.8)]
X_sourabh_test = X_sourabh[int(X_sourabh.shape[0]*0.8):]

X_sourav_train = X_sourav[:int(X_sourav.shape[0]*0.8)]
X_sourav_test = X_sourav[int(X_sourav.shape[0]*0.8):]

X_sudip_train = X_sudip[:int(X_sudip.shape[0]*0.8)]
X_sudip_test = X_sudip[int(X_sudip.shape[0]*0.8):]

# consolidated training and testing data for all users

X_train=X_arav_train.append([X_joel_train, X_christ_train, X_shubham_train, X_sourabh_train, X_sourav_train, X_sudip_train])
X_test=X_arav_test.append([X_joel_test, X_christ_test, X_shubham_test, X_sourabh_test, X_sourav_test, X_sudip_test])

y_train=X_train[['Class']]
y_test=X_test[['Class']]

# delete the index column in the dataFrame

X_data=X_train.append(X_test)
X_data = X_data.reset_index(drop=True)
y_data=y_train.append(y_test)
y_data = y_data.reset_index(drop=True)
print('\nPre-processing Done.')

print('\nCount of different classes in Train set:')
print(X_train['Class'].value_counts())

print('\nCount of different classes in Test set:')
print(X_test['Class'].value_counts())

# list of parameters to be used for trainig (excluding Class as it is the target variable)

feats=[c for c in X_train.columns if c!='Class']

# Train classifier

print('\nImplementing K-Nearest Neighbors Model.')
knn = KNeighborsClassifier(n_neighbors = 6  )
knn.fit(
    X_train[feats].values,
    y_train['Class']
)

# predicted labels for the test data

y_pred = knn.predict(X_test[feats].values)

# computing accuracy for the trainde model 

print("\nNumber of mislabeled points out of a total {} points : {}, Accuracy: {:05.5f}%"
      .format(
          X_test.shape[0],
          (X_test["Class"] != y_pred).sum(),
          100*(1-(X_test["Class"] != y_pred).sum()/X_test.shape[0])
))

#five fold cross validation

cv = KFold(n_splits=5)
clf = KNeighborsClassifier(n_neighbors = 6  )
X_data=X_data.values
y_data=y_data.values
accuracy=0
for traincv, testcv in cv.split(X_data):
        clf.fit(X_data[traincv], y_data[traincv])
        train_predictions = clf.predict(X_data[testcv])
        acc = accuracy_score(y_data[testcv], train_predictions)
        accuracy+= acc
       
accuracy = 20*accuracy
print('\n5 Fold Cross Validation Accuracy on Training Set: '+str(accuracy))
