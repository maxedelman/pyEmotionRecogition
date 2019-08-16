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

dset = pd.read_excel('/Users/MaxEdelman/Documents/featureDf.xlsx')
num_lists = int(len(dset.index) / 7)
lists = [[] for i in xrange(num_lists)]
listCounter = 0
counter = 0
dsetLists = dset.iloc[:, 1:].values

for row in dsetLists:
    if counter < 6:
        lists[listCounter].append(row)
        counter = counter + 1

    else:
        lists[listCounter].append(row)
        listCounter = listCounter + 1
        counter = 0
#print(lists)
def SVM(directory):
    dataset = pd.read_excel(directory)
    #testingdata = pd.read_excel('/Users/MaxEdelman/Documents/testingDf.xlsx')
    X = dataset.drop('emotion', axis=1)
    X = dataset.drop(dataset.columns[0], axis=1)
    y = dataset['emotion']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
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


def kernelSVM(directory):
    dataset = pd.read_excel (directory)
    testingdata = pd.read_excel ( '/Users/MaxEdelman/Documents/testingDf.xlsx' )
    X = dataset.drop('emotion', axis=1)
    X = dataset.drop(dataset.columns[0], axis=1)
    y = dataset['emotion']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    '''X_test = testingdata.drop('emotion', axis=1)
    X_test = testingdata.drop(testingdata.columns[0], axis=1)
    y_test = testingdata['emotion']'''
    svclassifier = SVC(kernel='poly', degree=8)
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)
    print(pd.crosstab(y_test, y_pred, rownames=['Actual emotion'], colnames=['Predicted emotion'] ))
    print(classification_report(y_test, y_pred))

def kNN(directory):
    print('knn for', directory)
    dataset = pd.read_excel (directory)
    X = dataset.iloc[:, 1:-1].values
    y = dataset.iloc[:, -1].values
    y = dataset['emotion']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    optimal_k = 21
    classifier = KNeighborsClassifier(n_neighbors= optimal_k)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print(y_pred)
    print(pd.crosstab(y_test, y_pred, rownames=['Actual emotion'], colnames=['Predicted emotion'] ))
    print(classification_report(y_test, y_pred))
    '''plt.plot(neighbors, MSE)
    plt.xlabel('Number of Neighbors K')
    plt.ylabel('Misclassification Error')
    plt.show()'''

def randForest(directory):
    print('random forest:', directory)
    dataset = pd.read_excel(directory)
    # Creating the dependent variable class
    factor = pd.factorize(dataset['emotion'])
    dataset.emotion = factor[0]
    definitions = factor[1]
    # Splitting the data into independent and dependent variables
    X = lists[:, :, :-1]
    y = lists[:, :, -1]
    print(y)
    # Creating the Training and Test set from data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=21)
    '''scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)'''
    scalers = {}
    for i in range (3):
        scalers[i] = StandardScaler ()
        X_train[:, i, :] = scalers[i].fit_transform ( X_train[:, i, :] )

    for i in range ( X_test.shape[1] ):
        X_test[:, i, :] = scalers[i].transform ( X_test[:, i, :] )
        # Fitting Random Forest Classification to the Training set
    classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=42)
    classifier.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    # Reverse factorize (converting y_pred from 0s,1s and 2s to Iris-setosa, Iris-versicolor and Iris-virginica
    reversefactor = dict(zip(range(4), definitions))
    '''y_test = np.vectorize(reversefactor.get)(y_test)
    y_pred = np.vectorize(reversefactor.get)(y_pred)'''
    print(pd.crosstab(y_test, y_pred, rownames=['Actual emotion'], colnames=['Predicted emotion'] ))
    print(classification_report(y_test, y_pred))
    plt.xlabel('features', size=14)
    plt.ylabel('emotion')
    plt.title('feature data vs. emotion', size=16)
    plt.plot(X_train, y_train,)
    plt.show()

def gaussianNB(directory):
    print('gaussian NB:')
    dataset = pd.read_excel (directory)
    # Creating the dependent variable class
    factor = pd.factorize ( dataset['emotion'] )
    dataset.emotion = factor[0]
    definitions = factor[1]
    # Splitting the data into independent and dependent variables
    X = dataset.iloc[:, 1:-1].values
    y = dataset.iloc[:, -1].values
    y1 = y + 1
    # Creating the Training and Test set from data
    X_train, X_test, y_train, y_test = train_test_split ( X, y1, test_size=0.30, random_state=21 )
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
    '''y_test = np.vectorize(reversefactor.get)(y_test)
    y_pred = np.vectorize(reversefactor.get)(y_pred)'''
    # Making the Confusion Matrix
    print(pd.crosstab ( y_test, y_pred, rownames=['Actual emotion'], colnames=['Predicted emotion'] ))
    print(classification_report ( y_test, y_pred ))
    plt.xlabel ( 'features', size=14 )
    plt.ylabel ( 'emotion' )
    plt.title ( 'GaussianNB', size=16 )
    plt.plot(X_train, y_train)
    plt.show ()
    print(y_train)
    print(y_test)
    print(y_pred)

def bernoulliNB(directory):
    print('Bernoulli NB:')
    dataset = pd.read_excel (directory)
    # Creating the dependent variable class
    factor = pd.factorize ( dataset['emotion'] )
    dataset.emotion = factor[0]
    definitions = factor[1]
    # Splitting the data into independent and dependent variables
    X = dataset.iloc[:, 1:-1].values
    y = dataset.iloc[:, -1].values
    y1 = y + 1
    # Creating the Training and Test set from data
    X_train, X_test, y_train, y_test = train_test_split ( X, y1, test_size=0.30, random_state=21 )
    # X_trainArray = X_train.values
    scaler = StandardScaler ()
    X_train = scaler.fit_transform ( X_train )
    X_test = scaler.transform ( X_test )
    # Fitting Random Forest Classification to the Training set
    classifier = BernoulliNB()
    classifier.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    # Reverse factorize (converting y_pred from 0s,1s and 2s to Iris-setosa, Iris-versicolor and Iris-virginica
    reversefactor = dict(zip(range(4), definitions))
    '''y_test = np.vectorize(reversefactor.get)(y_test)
    y_pred = np.vectorize(reversefactor.get)(y_pred)'''
    # Making the Confusion Matrix
    print(pd.crosstab ( y_test, y_pred, rownames=['Actual emotion'], colnames=['Predicted emotion'] ))
    print(classification_report ( y_test, y_pred ))




#SVM()
#kernelSVM()
#kNN('/Users/MaxEdelman/Documents/pcaDf.xlsx', 6)
randForest('/Users/MaxEdelman/Documents/featureDf.xlsx')
#gaussianNB('/Users/MaxEdelman/Documents/selectedFeatures.xlsx')
#bernoulliNB('/Users/MaxEdelman/Documents/selectedFeatures.xlsx')
