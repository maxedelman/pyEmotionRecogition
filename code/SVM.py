import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from mlxtend.plotting import plot_decision_regions
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB


def SVM():
    dataset = pd.read_excel('/Users/MaxEdelman/Documents/meanDf.xlsx')
    #testingdata = pd.read_excel('/Users/MaxEdelman/Documents/testingDf.xlsx')
    X = dataset.drop('emotion', axis=1)
    X = dataset.drop(dataset.columns[0], axis=1)
    y = dataset['emotion']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    X_trainArray = X_train.values
    '''X_test = testingdata.drop('emotion', axis=1)
    X_test = testingdata.drop(testingdata.columns[0], axis=1)
    y_test = testingdata['emotion']'''
    svclassifier = SVC(kernel='linear')
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)
    #print(confusion_matrix(y_test,y_pred))
    #print(classification_report(y_test,y_pred))
    plt.xlabel(X.columns[0], size=14)
    plt.ylabel('emotion')
    plt.title('SVM Decision Region Boundary', size=16)
    plt.plot(X_trainArray, y_train)
    plt.show()


def kernelSVM():
    dataset = pd.read_excel ( '/Users/MaxEdelman/Documents/meanDf.xlsx' )
    testingdata = pd.read_excel ( '/Users/MaxEdelman/Documents/testingDf.xlsx' )
    X = dataset.drop('emotion', axis=1)
    X = dataset.drop(dataset.columns[0], axis=1)
    y = dataset['emotion']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    '''X_test = testingdata.drop('emotion', axis=1)
    X_test = testingdata.drop(testingdata.columns[0], axis=1)
    y_test = testingdata['emotion']'''
    svclassifier = SVC(kernel='poly', degree=8)
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

def kNN():
    print('knn')
    dataset = pd.read_excel ( '/Users/MaxEdelman/Documents/meanDf.xlsx' )
    X = dataset.iloc[:, 1:35].values
    y = dataset.iloc[:, 35].values
    y = dataset['emotion']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    myList = list(range(1, 50))
    neighbors = filter(lambda x: x % 2 != 0, myList)
    #empty list that will hold cv scores
    cv_scores = []
    #perform 10-fold cross validation
    #finds optimal neighbors value
    for k in neighbors:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
        cv_scores.append(scores.mean())
    #changing to misclassification error
    MSE = [1 - x for x in cv_scores]
    #determining best k
    optimal_k = neighbors[MSE.index(min(MSE))]
    #print('optimal k is %d' % optimal_k)

    classifier = KNeighborsClassifier(n_neighbors= optimal_k)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    '''plt.plot(neighbors, MSE)
    plt.xlabel('Number of Neighbors K')
    plt.ylabel('Misclassification Error')
    plt.show()'''

def randForest(start, end):
    print('random forest:')
    dataset = pd.read_excel('/Users/MaxEdelman/Documents/meanDf.xlsx')
    # Creating the dependent variable class
    factor = pd.factorize(dataset['emotion'])
    dataset.emotion = factor[0]
    definitions = factor[1]
    # Splitting the data into independent and dependent variables
    X = dataset.iloc[:, start:end].values
    y = dataset.iloc[:, 35].values
    # Creating the Training and Test set from data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=21)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # Fitting Random Forest Classification to the Training set
    classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=42)
    classifier.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    # Reverse factorize (converting y_pred from 0s,1s and 2s to Iris-setosa, Iris-versicolor and Iris-virginica
    reversefactor = dict(zip(range(4), definitions))
    y_test = np.vectorize(reversefactor.get)(y_test)
    y_pred = np.vectorize(reversefactor.get)(y_pred)
    print('y pred is: ', y_pred)
    print(pd.crosstab(y_test, y_pred, rownames=['Actual emotion'], colnames=['Predicted emotion'] ))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    plt.xlabel('features', size=14)
    plt.ylabel('emotion')
    plt.title('randForest', size=16)
    plt.plot(X_train, y_train,)
    plt.show()

def gaussianNB():
    print('gaussian NB:')
    dataset = pd.read_excel ( '/Users/MaxEdelman/Documents/meanDf.xlsx' )
    # Creating the dependent variable class
    factor = pd.factorize ( dataset['emotion'] )
    dataset.emotion = factor[0]
    definitions = factor[1]
    # Splitting the data into independent and dependent variables
    X = dataset.iloc[:, 1:35].values
    y = dataset.iloc[:, 35].values
    # Creating the Training and Test set from data
    X_train, X_test, y_train, y_test = train_test_split ( X, y, test_size=0.25, random_state=21 )
    # X_trainArray = X_train.values
    scaler = StandardScaler ()
    X_train = scaler.fit_transform ( X_train )
    X_test = scaler.transform ( X_test )
    # Fitting Random Forest Classification to the Training set
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    # Reverse factorize (converting y_pred from 0s,1s and 2s to Iris-setosa, Iris-versicolor and Iris-virginica
    reversefactor = dict(zip(range(4), definitions))
    y_test = np.vectorize(reversefactor.get)(y_test)
    y_pred = np.vectorize(reversefactor.get)(y_pred)
    y_train = y_train + 1
    # Making the Confusion Matrix
    print(pd.crosstab ( y_test, y_pred, rownames=['Actual emotion'], colnames=['Predicted emotion'] ))
    print(confusion_matrix ( y_test, y_pred ))
    print(classification_report ( y_test, y_pred ))
    plt.xlabel ( 'features', size=14 )
    plt.ylabel ( 'emotion' )
    plt.title ( 'GaussianNB', size=16 )
    plt.plot(X_train, y_train)
    plt.show ()



#SVM()
#kernelSVM()
#kNN()
randForest(9, 22)
#gaussianNB()
