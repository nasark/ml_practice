import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier


# Calculates test error by calculating mean squared error between prediction and test set
def calcTestError(y, t):
    error=0
    for i in range(np.size(y)):                 #summation
        error = error + (y[i]-t[i])**2          #error = (1/N)*(y-t)^2

    error = error/np.size(y)        #(1/N)*error summation
    return error

# Decision tree classifier which uses cross validation to find the best max number of leaves between 2-400
def decisionTree(X_train, X_test, t_train, t_test):
    scores = []
    cv_errors = []
    # Create and train models from max leaves between 2-400, then store their scores and cross validation errors obtained on the test set
    for i in range(2, 401):
        # create and train model with max leaves=i
        model = DecisionTreeClassifier(max_leaf_nodes=i)
        model.fit(X_train, t_train)
        #obtain scores of model using cross validation
        score = cross_val_score(model, X_test, t_test)
        # obtain test error
        cv_errors.append(1-score.mean())
        #final score of model
        scores.append(score.mean())
    
    # obtain score of best model based on position in list
    best_score_pos = scores.index(max(scores))
    # max number of leaves is equal to position of the best score model + 2 since iteration was from 2-400 leaves
    best_max_leaves = best_score_pos+2
    # recreate and train model with best max leaves
    best_model = DecisionTreeClassifier(max_leaf_nodes=best_max_leaves)
    best_model.fit(X_train, t_train)
    # obtain prection and test error
    y_pred = best_model.predict(X_test)
    test_error = calcTestError(y_pred, t_test)

    return test_error, cv_errors

# Bagging classifier - returning the test errors for 5 bagging classifiers (down from 50 due to slow run time) 
def bagging(X_train, X_test, t_train, t_test):
    test_errors = []
    # iterate from 200 predictors to 1000 at intervals of 200
    for i in range(200, 1200, 200):
        # create and train models with predictors from 200 to 1000 at intervals of 200
        model = BaggingClassifier(n_estimators=i)
        model.fit(X_train, t_train)
        # obtain prediction for model and store test error
        y_pred = model.predict(X_test)
        test_errors.append(calcTestError(y_pred, t_test))
    
    return test_errors

# Random forest classifier with decision trees of no restriction - returning the test errors for 5 random forest classifiers (down from 50 due to slow run time) 
def randomForest(X_train, X_test, t_train, t_test):
    test_errors = []
    # iterate from 200 predictors to 1000 at intervals of 200
    for i in range(200, 1200, 200):
        # create and train models with predictors from 200 to 1000 at intervals of 200
        model = RandomForestClassifier(n_estimators=i)
        model.fit(X_train, t_train)
        # obtain prediction for model and store test error
        y_pred = model.predict(X_test)
        test_errors.append(calcTestError(y_pred, t_test))
    
    return test_errors

# Adaboost classifier - returning the test errors for 5 Adaboost classifiers (down from 50 due to slow run time) 
def adaBoost(X_train, X_test, t_train, t_test, leaf_num):
    test_errors = []
    # iterate from 200 predictors to 1000 at intervals of 200
    for i in range(200, 1200, 200):
        # create and train models with predictors from 200 to 1000 at intervals of 200
        # if leaf_num=0, then decision stumps will be used
        # if leaf_num=10, then decision trees with a maximum of 10 leaves will be used
        # otherwise decision trees with no restrictions will be used
        if (leaf_num==0):
            model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=i)
        elif (leaf_num==10):
            model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_leaf_nodes=leaf_num), n_estimators=i)
        else:
            model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=i)
        model.fit(X_train, t_train)
        # obtain prediction for model and store test error
        y_pred = model.predict(X_test)
        test_errors.append(calcTestError(y_pred, t_test))
    
    return test_errors


# Initialize data
dataset = pd.read_csv('spambase.data')
X = dataset.iloc[:, :-1].values
t = dataset.iloc[:, -1].values
X_train, X_test, t_train, t_test = train_test_split(X, t, test_size = 1/3, random_state = 7997)

# obtain test errors for different classifiers as specified in specification
test_error, cv_errors = decisionTree(X_train, X_test, t_train, t_test)
test_error1 = [test_error, test_error, test_error, test_error, test_error]  # used for plot of decision tree classifier
test_error2 = bagging(X_train, X_test, t_train, t_test)
test_error3 = randomForest(X_train, X_test, t_train, t_test)
test_error4 = adaBoost(X_train, X_test, t_train, t_test, 0)
test_error5 = adaBoost(X_train, X_test, t_train, t_test, 10)
test_error6 = adaBoost(X_train, X_test, t_train, t_test, 1)

# plots
plt.plot(range(200, 1200, 200), test_error1, c='red', label='Decision Tree')
plt.plot(range(200, 1200, 200), test_error2, c='blue', label='Bagging')
plt.plot(range(200, 1200, 200), test_error3, c='green', label='Random Forest')
plt.plot(range(200, 1200, 200), test_error4, c='orange', label='AdaBoost w/ stump')
plt.plot(range(200, 1200, 200), test_error5, c='purple', label='AdaBoost w/ 10 leaves')
plt.plot(range(200, 1200, 200), test_error6, c='cyan', label='AdaBoost w/ unlimited leaves')
plt.legend()
plt.title('Test Errors vs. Number of Predictors')
plt.xlabel('Number of Predictors')
plt.ylabel('Test Error')
fig4 = plt.figure()

plt.plot(range(2, 401), cv_errors, c='red')
plt.title('Cross Validation Errors vs. Number of Leaves for Decision Tree Classifier')
plt.xlabel('Number of Leaves')
plt.ylabel('Cross Validation Error')

plt.show()