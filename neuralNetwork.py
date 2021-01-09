import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ReLU - activation function at hidden layers
def ReLU(z):
    return np.maximum(0, z)

# derivative of ReLU function - used in back propagation
def dReLU(z):
    for i in range(np.size(z, 0)):
        for j in range(np.size(z, 1)):
            if z[i][j] <= 0:
                z[i][j] = 0
            else:
                z[i][j] = 1
    
    return z

# sigmoid - activation function at output layer
def sigmoid(z):
    return 1/(1+np.exp(-z))

# derivative of sigmoid function - used in back propagation
def dSigmoid(z):
    s = 1/(1+np.exp(-z))
    return s*(1-s)

# calculates and returns cross-entropy loss - -1/N*(tlog(y) + (1-t)log(1-y))
def crossEntropyLoss(prediction, target):
    sum = 0
    prediction = prediction.T
    for i in range(np.size(prediction)):
        sum = sum + target[i]*np.log(prediction[i]) + (1-target[i])*np.log(1-prediction[i])
    error = (-1/np.size(prediction))*sum

    return error

# derivative of cross entropy loss - used in back propagation, -t + sigmoid(z3)
def dCrossEntropyLoss(prediction, target):
    return -target + sigmoid(prediction)

# binary classifier for predictions made at output layer, theta=0
def binaryClassifier(prediction):
    y = np.zeros(np.shape(prediction))
    for i in range(np.size(prediction)):
        if(prediction[i]>=0):           # if prediction>=0, classify as class 1, otherwise class 0
            y[i] = 1
    return y

# calculates misclassification rate
def calcMisclassification (prediction, target):
    prediction = prediction.T
    target = target.reshape((target.shape[0], 1))
    # obtain classifications based on predictions
    y = binaryClassifier(prediction)
    miss=0
    for i in range(np.size(prediction)):
        # count mismatches
        if (y[i] != target[i]):
            miss += 1
    error = miss/np.size(prediction)  # number of misclassified examples/total examples

    return error

# feed forward process from input layer to output layer
def feedForward(X_train, t_train, w1, w2, w3):
    # calculations between input layer and first hidden layer
    z1 = np.dot(w1, X_train) 
    h1 = ReLU(z1)
    # add row of ones to top of h1
    row_ones = np.ones((1, np.size(h1, 1)))
    h1 = np.vstack((row_ones, h1))
    # calculations between first hidden layer and second hidden layer
    z2 = np.dot(w2, h1) 
    h2 = ReLU(z2)
    # add row of ones to top of h2
    row_ones = np.ones((1, np.size(h2, 1)))
    h2 = np.vstack((row_ones, h2))
    # calculations between second hidden layer and output layer - predictions achieved
    z3 = np.dot(w3, h2) 
    output = sigmoid(z3)

    # back propagation to determine change in weights
    dw1, dw2, dw3 = backPropagation(X_train, t_train, w2, w3, h1, h2, z1, z2, z3)

    return output, dw1, dw2, dw3

# back propagation process to determine change in weights
def backPropagation(X_train, t_train, w2, w3, h1, h2, z1, z2, z3):
    # layer 3 (output layer)
    h2 = h2.T
    dz3 = dCrossEntropyLoss(z3, t_train)    # dz3 = -t + sigmoid(z3)
    dw3 = np.dot(dz3, h2)        #dw3 = dz3*[1 h2.T]
    w3 = w3[: , 1:]     #w3_bar
    w3 = w3.T           #w3 without first column and transposed -> w3_bar.T
    dz2 = np.multiply(dReLU(z2), np.dot(w3, dz3))   #dz2 = dReLU(z2)*(w3_bar*dz3) element-wise multiplication between dReLU(z2) and dot(w3, dz3)

    # layer 2 (second hidden layer)
    h1 = h1.T
    dw2 = np.dot(dz2, h1)   #dw2 = dz2*[1 h1.T]
    w2 = w2[: , 1:]     #w2_bar
    w2 = w2.T           #w2 without first column and transposed -> w2_bar.T
    dz1 = np.multiply(dReLU(z1), np.dot(w2, dz2))    #dz1 = dReLU(z1)*(w2_bar*dz2) element-wise multiplication between dReLU(z1) and dot(w2, dz2)

    # layer 1 (first hidden layer)
    dw1 = np.dot(dz1, X_train.T)    #dw1 = dz1*[1 X.T]

    return dw1, dw2, dw3

# same feed forward process but for the validation set
# back propagation isn't needed since weights are affected by training set only also the predictions are needed from the validation set
def feedForwardValid(X_valid, w1, w2, w3):
    z1 = np.dot(w1, X_valid) 
    h1 = ReLU(z1)

    row_ones = np.ones((1, np.size(h1, 1)))
    h1 = np.vstack((row_ones, h1))

    z2 = np.dot(w2, h1) 
    h2 = ReLU(z2)
    row_ones = np.ones((1, np.size(h2, 1)))
    h2 = np.vstack((row_ones, h2))

    z3 = np.dot(w3, h2) 
    output = sigmoid(z3)

    return output

# shuffles columns of train set at the start of an epoch during SGD
def shuffle(X_train):
    X_train = X_train.T
    np.random.shuffle(X_train)
    X_train = X_train.T

    return X_train

# SGD process
def stochasticGradientDescent(X_train, t_train, X_valid, t_valid, n1, n2):
    lr = 0.0001
    # add dummy row for biases
    # transpose so that each row represents a feature
    X_train = X_train.T
    bias_row = np.ones((1, np.size(X_train, 1)))
    X_train = np.vstack((bias_row, X_train))
    X_valid = X_valid.T
    bias_row = np.ones((1, np.size(X_valid, 1)))
    X_valid = np.vstack((bias_row, X_valid))

    #weights initialized using random number generator
    # w1 matrix has dimensions (nodes at hidden layer 1, nodes at input layer)
    w1 = np.random.rand(n1, X_train.shape[0])
    # w2 matrix has dimensions (nodes at hidden layer 2, nodes at hidden layer 1 + 1 for bias)
    w2 = np.random.rand(n2, n1+1)
    # w3 matrix has dimensions (nodes at output layer, nodes at hidden layer 2 + 1 for bias)
    w3 = np.random.rand(1, n2+1)

    #lists to keep track of weights, errors, and misclassification rates throughout epochs
    w1_arr = []
    w1_arr.append(w1)
    w2_arr = []
    w2_arr.append(w2)
    w3_arr = []
    w3_arr.append(w3)

    training_errors = []
    training_missrates = []
    validation_errors = []
    validation_missrates = []

    for epoch in range(100):
        #shuffle training set
        X_train = shuffle(X_train)

        #retrieve predictions from training and validation sets
        training_output, dw1, dw2, dw3 = feedForward(X_train, t_train, w1, w2, w3)
        validation_output = feedForwardValid(X_valid, w1, w2, w3)

        #retrieve cross entropy errors and misclassification rates for training and validation sets
        training_error = crossEntropyLoss(training_output, t_train)
        training_miss = calcMisclassification(training_output, t_train)

        validation_error = crossEntropyLoss(validation_output, t_valid)
        validation_miss = calcMisclassification(validation_output, t_valid)

        #update weights
        w1 -= lr*dw1
        w2 -= lr*dw2
        w3 -= lr*dw3

        #add weights and errors to respective lists
        w1_arr.append(w1)
        w2_arr.append(w2)
        w3_arr.append(w3)

        training_errors.append(training_error)
        training_missrates.append(training_miss)

        validation_errors.append(validation_error)
        validation_missrates.append(validation_miss)
    
    #get position of lowest validation error
    best_pos = validation_errors.index(min(validation_errors))
    # return errors and best weights determined by lowest validation error
    return training_errors, training_missrates, validation_errors, validation_missrates, w1_arr[best_pos], w2_arr[best_pos], w3_arr[best_pos]

# main
# Initialize data and splits
dataset = pd.read_csv('data_banknote_authentication1.txt')
X = dataset.iloc[:, :-1].values
t = dataset.iloc[:, -1].values

X_train, X_test, t_train, t_test = train_test_split(X, t, test_size = 1/2, random_state = 7997)
X_test, X_valid, t_test, t_valid = train_test_split(X_test, t_test, test_size = 1/2, random_state = 7997)
sc = StandardScaler()
X_train[:, :]  = sc.fit_transform(X_train[:, :])
X_valid[:, :]  = sc.transform(X_valid[:, :])
X_test[:, :]  = sc.transform(X_test[:, :])

# model 1 - first two features
# output for all 9 possible values of (n1, n2) [2, 3, 4]
# displays a plot of the errors for each of 9 possibilites
for n1 in range(2, 5):
    for n2 in range(2, 5):
        training_errors, training_missrates, validation_errors, validation_missrates, w1, w2, w3 = stochasticGradientDescent(X_train[:, :2], t_train, X_valid[:, :2], t_valid, n1, n2)
        
        plt.plot(range(100), training_errors, c='blue', label='Training Cross-Entropy')
        plt.plot(range(100), training_missrates, c='green', label='Training Misclassification')
        plt.plot(range(100), validation_errors, c='red', label='Validation Cross-Entropy')
        plt.plot(range(100), validation_missrates, c='purple', label='Validation Misclassification')
        plt.title('Errors vs. Epoch, n1=' + str(n1) + ', n2=' + str(n2))
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        
        print('For D=2, n1=' + str(n1) + ', and n2=' + str(n2))
        print('Lowest Validation error ', min(validation_errors))
        print('W1', w1)
        print('W2', w2)
        print('W3', w3)

        plt.show()

# model 2 - first three features
for n1 in range(2, 5):
    for n2 in range(2, 5):
        training_errors, training_missrates, validation_errors, validation_missrates, w1, w2, w3 = stochasticGradientDescent(X_train[:, :1], t_train, X_valid[:, :1], t_valid, n1, n2)

        print('For D=3, n1=' + str(n1) + ', and n2=' + str(n2))
        print('Lowest Validation error ', min(validation_errors))
        print('W1', w1)
        print('W2', w2)
        print('W3', w3)

# model 3 - all features
for n1 in range(2, 5):
    for n2 in range(2, 5):
        training_errors, training_missrates, validation_errors, validation_missrates, w1, w2, w3 = stochasticGradientDescent(X_train, t_train, X_valid, t_valid, n1, n2)

        print('For D=4, n1=' + str(n1) + ', and n2=' + str(n2))
        print('Lowest Validation error ', min(validation_errors))
        print('W1', w1)
        print('W2', w2)
        print('W3', w3)
