import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold

# Calculate error between y (prediction) and t (validation set)
def calcError(y, t):
    error=0
    for i in range(np.size(y)):                 #summation
        error = error + (y[i]-t[i])**2          #error = (1/N)*(y-t)^2

    error = error/np.size(y)        #(1/N)*error summation
    return error

# Train model and return parameter vector w
def trainModel(X_train, t_train):
    # Model is trained using linear regression: w = inv(XT*X)*XT*t
    A = np.dot(X_train.T, X_train)      #A = XT*X
    A_inv = np.linalg.inv(A)
    B = np.dot(X_train.T, t_train)      #B = XT*t
    return np.dot(A_inv, B)

# Perform K-fold cross validation for a selected feature (included in S) for some value of k
def kFoldCrossValidation (S, Y):
    split_error = 0
    kfold = KFold(n_splits=5, random_state=7997, shuffle=True)  #Initialize Kfold with K=5 and last 4 digits of student number
    for train, test in kfold.split(S):
        S_train, S_test = S[train], S[test]     #split train and target sets accordingly
        Y_train, Y_test = Y[train], Y[test]     
        w = trainModel(S_train, Y_train)
        y = np.dot(S_test, w) # calulate prediction
        split_error = split_error + calcError(y, Y_test)    #calculate error for this split and accumulate total error for all splits
    return split_error                                      #this error will then be divided by 5 to get the average which is the cross validation error

# Construct set S by using the greedy algorithm
def constructS (data, power):           #power parameter is used to implement basis function - if power=1 no basis function is being used
    X, Y = data[:,:], data[:,-1]        # initialize train and target sets from data
    kfold_error = []
    test_error = []
    S = np.ones((np.size(X, 0),1))      # initialize S as dummy vector of 1s
    for k in range(1,14):               # loop from k=1-13
        feature_error = []
        for i in range(1,np.size(X, 1)+1):      # loops as many times as features there are in X
            S = np.hstack((S, X[:,i-1:i]))      # temporarily add feature to S
            S2 = np.power(S, power)             # if power != 1, a basis function is being applied to S i.e. power=1/2 means sqrt(x) is basis function
            Y2 = np.power(Y, power)
            split_error = kFoldCrossValidation(S2, Y2)  
            feature_error.append(split_error/5)         # cross fold error obtained by averaging accumulated error from splits - stored in list
            S = np.delete(S, np.size(S, 1)-1, 1)  # reset S by removing temporarily added feature
        if (feature_error):
            feat_index = feature_error.index(min(feature_error))    # obtain index of feature with lowest cross fold error
            print ("feature %i" %k)
            print(feat_index)
            print(feature_error)
            kfold_error.append(feature_error[feat_index])           # store lowest cross fold error
        else:
            break
        S = np.hstack((S, X[:,feat_index:feat_index+1]))        # add feature with lowest cross fold error to S
        X = np.delete(X, feat_index, 1)                         # remove that feature from X so that it is not included in further iterations of greedy algorithm
        w = trainModel(S, Y)                                
        y_pred = np.dot(S, w)                               # train model with new feature added to S in order to obtain test error
        test_error.append(calcError(y_pred, Y))             # store test error
    if (power == 1):                                            # return S, array of cross validation errors for selected features in S, and respective test errors
        return S[:, 1:14], kfold_error, test_error              #S[:,1:14] is returned because the dummy vector of 1s is no longer needed
    else:
        return np.power(S[:, 1:14], power), kfold_error, test_error     #if basis function is being used


# main for part 2
data = load_boston().data
#list indices for kfold_error and test_error indicate the respective k value
S, kfold_error, test_error = constructS(data, 1)        #no basis function
S2, kfold_error2, test_error2 = constructS(data, 1/2)   #basis function = sqrt(x) -> lower cross fold errors than no basis function
S3, kfold_error3, test_error3 = constructS(data, 2)     #basis function = x^2

# print errors to console 
print("No basis expansion")
print(kfold_error)
print(test_error)

print("sqrt(x)")
print(kfold_error2)
print(test_error2)

print("x^2")
print(kfold_error3)
print(test_error3)

# plots for errors
plt.plot(range(1,14), kfold_error, c='red', label='kfold error')
plt.plot(range(1,14), test_error, c='blue', label='test error')
plt.legend()
plt.title('Error vs. k (No Basis Expansion)')
plt.xlabel('k')
plt.ylabel('Error')
fig1 = plt.figure()

plt.plot(range(1,14), kfold_error2, c='red', label='kfold error')
plt.plot(range(1,14), test_error2, c='blue', label='test error')
plt.legend()
plt.title('Error vs. k (Basis Expansion = sqrt(x))')
plt.xlabel('k')
plt.ylabel('Error')
fig2 = plt.figure()

plt.plot(range(1,14), kfold_error3, c='red', label='kfold error')
plt.plot(range(1,14), test_error3, c='blue', label='test error')
plt.legend()
plt.title('Error vs. k (Basis Expansion = x^2')
plt.xlabel('k')
plt.ylabel('Error')
fig3 = plt.figure()

plt.plot(range(1,14), kfold_error, c='red', label='kfold error no basis expansion')
plt.plot(range(1,14), test_error, c='blue', label='test error no basis expansion')
plt.plot(range(1,14), kfold_error2, c='green', label='kfold error sqrt(x)')
plt.plot(range(1,14), test_error2, c='cyan', label='test error sqrt(x)')
plt.plot(range(1,14), kfold_error3, c='magenta', label='kfold error x^2')
plt.plot(range(1,14), test_error3, c='yellow', label='test error x^2')
plt.legend()
plt.title('Error vs. k')
plt.xlabel('k')
plt.ylabel('Error')

plt.show()

