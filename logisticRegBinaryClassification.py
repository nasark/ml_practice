import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# Sigmoid function for logistic regression
def sigmoid(z):
    return 1/(1 + np.exp(-z))

# Implements gradient descent to calculate w
def gradientDescent(alpha, iter, X, t):
    w = np.zeros(X.shape[1])
    z = np.zeros(len(X))

    for i in range(iter):
        z = np.dot(X, w)
        y = sigmoid(z)
        gradient = np.dot(X.T, (y-t))/len(t) 
        w = w - alpha * gradient      # update w every iteration
     
    return w

# Implements classifier function for given theta threshold value to determine predictions
def classifier(z, theta, M):
    y = np.zeros(M)
    for i in range(M):
        if (z[i]>=theta):   # if z >= theta prediction is 1, otherwise it will remain 0
            y[i] = 1
    return y

# Calculates precision and recall metrics based on prediction and test set
def calcPR(y, t_test):
    # Initialize values for truly positives (tp), false positives (fp), and false negatives (fn)
    tp = 0
    fp = 0
    fn = 0
    for i in range(len(t_test)):
        # iterate through test set and accumulate values for tp, fp, and fn based values of prediction and test set
        if (y[i]==1 and t_test[i]==1):
            tp+=1
        elif (y[i]==1 and t_test[i]==0):
            fp+=1
        elif (y[i]==0 and t_test[i]==1):
            fn+=1
    # calculate precision and recall
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    return precision, recall

# Calculates misclassification rate
def calcMisclassification(z, t_test, M):
    y = classifier(z, 0, M)         # obtain predictions to minimize misclassification rate using classifier with theta=0
    miss = y - t_test
    err = np.count_nonzero(miss)/M  # number of misclassified examples/total examples

    return err

# F1 score is calculated using the formula: F1 = 2PR/(P+R)
def calcF1(precision, recall):
    precision_mean = sum(precision)/len(precision)
    recall_mean = sum(recall)/len(recall)

    return (2*precision_mean*recall_mean)/(precision_mean + recall_mean)

# Compute precision, recall, F1, and test error for logistic regression model
def computeMetrics(z, theta, M, t_test):
    precision = np.zeros(M)
    recall = np.zeros(M)
    for i in range(M):                  # precision and recall are calculated using predictions from classifiers with threshold values of theta
        y = classifier(z, theta[i], M)      #theta consists of values of z, sorted in order
        precision[i], recall[i] = calcPR(y, t_test)     # precision and recall values are calculated and stored for theta value

    F1 = calcF1(precision, recall)
    test_error = calcMisclassification(z, t_test, M)

    return precision, recall, F1, test_error

# Logistic regression implementation
def logisticReg(X_train, X_test, t_train, t_test):
    # using only the last two features for predictions
    X_train = X_train[:,2:]
    X_test = X_test[:,2:]

    N = len(X_train)
    M = len(X_test)

    # add dummy vector to test and train set
    dummy = np.ones(N)
    X1_train = np.insert(X_train, 0, dummy, axis=1)
    dummy = np.ones(M)
    X1_test = np.insert(X_test, 0, dummy, axis=1)

    # perform gradient descent to obtain vector w, alpha = 1 and 300 iterations
    w = gradientDescent(1, 300, X1_train, t_train)
    # obtain prediction z = wTx = xw
    z = np.dot(X1_test,w)
    # the values of theta (thresholds for classifiers), are the values of z sorted
    theta = np.asarray(z).flatten().tolist()
    theta.sort()

    # obtain metrics for model
    precision, recall, F1, test_error = computeMetrics(z, theta, M, t_test)

    return precision, recall, F1, test_error

# Logistic regression using Scikit learn
def logisticRegScikit(X_train, X_test, t_train, t_test):
    N = len(X_train)
    M = len(X_test)

    # add dummy vector to test and train set
    dummy = np.ones(N)
    X1_train = np.insert(X_train, 0, dummy, axis=1)
    dummy = np.ones(M)
    X1_test = np.insert(X_test, 0, dummy, axis=1)

    # obtain and train model
    model = LogisticRegression(random_state=7997)
    model.fit(X1_train, t_train)

    # obtain prediction z = wTx = xw
    z = model.predict_proba(X1_test)[:,1]
    # the values of theta (thresholds for classifiers), are the values of z sorted
    theta = np.asarray(z).flatten().tolist()
    theta.sort()

    # obtain metrics for model
    precision, recall, F1, test_error = computeMetrics(z, theta, M, t_test)

    return precision, recall, F1, test_error

# Perform K-fold cross validation for a selected feature for some value of k (1-5)
def kFoldCrossValidationKnn (X, Y, k):
    cv_error = []
    kfold = KFold(n_splits=5, random_state=7997, shuffle=True)  #Initialize Kfold with K=5 and last 4 digits of student number
    for i in range(1, k+1):
        split_error = 0
        for train, test in kfold.split(X):
            y_pred = []
            X_train, X_test = X[train], X[test]     #split train and target sets accordingly
            Y_train, Y_test = Y[train], Y[test] 
            for j in X_test:
                y_pred.append(Knn(X_train, j, Y_train, i))      # obtain predictions for k nearest neighbors
            split_error = split_error + calcError(y_pred, Y_test)    #calculate error for this split and accumulate total error for all splits
        cv_error.append(split_error/5)
    return cv_error                                     #this error will then be divided by 5 to get the average which is the cross validation error

# Calculate error between y (prediction) and t (test set) using mean square error
def calcError(y, t):
    error=0
    for i in range(np.size(y)):                 #summation
        error = error + (y[i]-t[i])**2          #error = (1/N)*(y-t)^2

    error = error/np.size(y)        #(1/N)*error summation
    return error

# K nearest neighbors implementation
def Knn(X_train, X_test, t_train, k):
    # calculate euclidean distance between points
    diff = (X_train - X_test)**2        
    distances = np.sqrt(diff.sum(axis=0))
    # sort distances and retrieve until k
    sorted_dist = distances.argsort()[:k]

    count = {}
    # number of points belonging to each class are counted
    for i in sorted_dist:
        label = t_train[i]
        count[label] = count.get(label, 0) + 1

    # return prediction which is the class with the most data points
    return max(count.values())


# K nearest neighbors with cross validation using Scikit learn
def KnnScikit(X_train, X_test, t_train, t_test):
    scores = []
    k_error = []
    # Create and train models from k=1-5, then store their scores obtained on the test set
    for i in range(1,6):
        model = KNeighborsClassifier(n_neighbors=i)
        model.fit(X_train, t_train)
        k_error.append(calcError(model.predict(X_test), t_test))    # store error for this k
        score = cross_val_score(model, X_test, t_test)
        scores.append(score.mean())

    # obtain score of best model based on position in list
    best_score_pos = scores.index(max(scores))
    # recreate model with best score and respective k value
    best_model = KNeighborsClassifier(n_neighbors=best_score_pos)
    # train model
    best_model.fit(X_train, t_train)
    # obtain prediction
    y_pred = best_model.predict(X_test)
    # calculate test error
    test_error = calcError(y_pred, t_test)

    return test_error, k_error          #return test error for best model and errors associated with k values (1-5)

#main for part 1

# Standardize features in training and test sets
sc = StandardScaler()
X, t = load_breast_cancer(return_X_y=True)
X_train, X_test, t_train, t_test = train_test_split(X, t, test_size = 1/4, random_state = 7997)
X_train[:,:] = sc.fit_transform(X_train[:,:])
X_test[:,:] = sc.transform(X_test[:,:])


# obtain metrics for logistic regression models using current implmentation and scikit learn
precision, recall, F1, test_error = logisticReg(X_train, X_test, t_train, t_test)
precision2, recall2, F12, test_error2 = logisticRegScikit(X_train, X_test, t_train, t_test)

# plot metrics and print F1 scores, misclassification rates to console
print("F1 Score for Logistic Regression Implementation: %f" %F1)
print("Misclassification rate for Logistic Regression Implementation: %f" %test_error)
print("F1 Score for Scikit-Learn Logistic Regression: %f" %F12)
print("Misclassification rate for Scikit-Learn Logistic Regression: %f" %test_error2)
plt.plot(recall, precision, c='blue', label='Implementation')
plt.title('Precision vs. Recall')
plt.plot(recall2, precision2, c='green', label='Scikit Learn')
plt.legend()
plt.xlabel('Recall')
plt.ylabel('Precision')
fig1 = plt.figure()

plt.plot(recall, precision, c='blue')
plt.title('Precision vs. Recall (Implementation)')
plt.xlabel('Recall')
plt.ylabel('Precision')
fig2 = plt.figure()

plt.plot(recall2, precision2, c='green')
plt.title('Precision vs. Recall (Scikit-Learn)')
plt.xlabel('Recall')
plt.ylabel('Precision')
fig3 = plt.figure()

# print test error obtained from k nearest neighbors implementation to console
knn_error = kFoldCrossValidationKnn(X, t, 5)
print(knn_error)

# print test errors obtained from k nearest neighbors scikit implementation to console
knn_error2, k_errors = KnnScikit(X_train, X_test, t_train, t_test)
print(knn_error2)
print(k_errors)