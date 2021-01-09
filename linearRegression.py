import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold

def dataGen(M):
    # Generate data sets
    X_train = np.linspace(0.,1.,10) # training set
    X_valid = np.linspace(0.,1.,100) # validation set
    np.random.seed(7997)
    t_valid = np.sin(4*np.pi*X_valid) + 0.3 * np.random.randn(100)
    t_train = np.sin(4*np.pi*X_train) + 0.3 * np.random.randn(10) 

    # Construct matrix X for training and validation sets - basis expansion
    X1_train = np.ones(10).reshape(-1,1)    #first column in predictor is simply 1s because of w0
    X1_valid = np.ones(100).reshape(-1,1)
    if (M>0):
        for i in range(1, M+1):     #X must have D+1 columns (number of features + 1)
            X1_col = X_train.reshape(-1,1)
            X1_col = np.power(X1_col, i)    #column vector is raised to a certain power depending on column number due to predictor formula
            X1_valid_col = X_valid.reshape(-1,1)
            X1_valid_col = np.power(X1_valid_col, i)    #repeat for validation set
            X1_train = np.hstack((X1_train, X1_col))    #append calculated column vector to matrix X
            X1_valid = np.hstack((X1_valid, X1_valid_col))

    print("Training")
    f_M, train_error = train(M, X1_train, t_train)
    print("Validation")
    f_M_valid, valid_error = train(M, X1_valid, t_valid)
    print("Average")
    avg_error = calcError(f_M_valid, 0, np.sin(4*np.pi*X_valid), 0)

    # Generate plots
    # Curves/plots for training set
    plt.scatter(X_train, t_train, color='b', marker='o', label='training data set')  #scatter plot to show data points
    plt.plot(X_valid, np.sin(4*np.pi*X_valid), c='green', label='f_true')  #f_true
    if (M==9):
        plt.plot(X_train, f_M, c='red', label='f_M w/ Training Examples, lambda1=1.53e-8')  #f_M with training examples
    else:
        plt.plot(X_train, f_M, c='red', label='f_M w/ Training Examples')  #f_M with training examples
    plt.legend()
    plt.title('Curve Fitting for Training Data Set with M=%i' %M)
    plt.xlabel('x')
    plt.ylabel('t')
    fig1 = plt.figure()

    # Curves/plots for validation set
    plt.scatter(X_valid, t_valid, color='b', marker='o', label='validation data set') #scatter plot to show data points
    plt.plot(X_valid, np.sin(4*np.pi*X_valid), c='green', label='f_true')  #f_true
    if (M==9):
        plt.plot(X_valid, f_M_valid, c='red', label='f_M w/ Validation Examples, lambda1=1.53e-8')  #f_M with validation examples 
    else:
        plt.plot(X_valid, f_M_valid, c='red', label='f_M w/ Validation Examples')  #f_M with validation examples
    plt.legend()
    plt.title('Curve Fitting for Validation Data Set with M=%i' %M)
    plt.xlabel('x')
    plt.ylabel('t')

    plt.show()

    return train_error, valid_error, avg_error

# Train using least squares linear regression
def train(M, X, t):
    # For M=9, need to use lambda and therefore construct matrix B
    if (M==9):
        lambda1 = 1.53e-8
        B = np.zeros((10,10)) 
        np.fill_diagonal(B, 2*lambda1)   #all non-diagonal elements are 0, diagonal elements are equal to 2*lambda, and first element=0
        B[0,0] = 0
    # Calculating w for training set, w = inv(XT*X)*XT*t or w=inv(XT*X + (N/2)*B)*Xt*t if M=9
    if (M==9):
        A = np.dot(X.T, X) + (np.linalg.matrix_rank(X)/2)*B  #Need to use regularization if M=9 
    else:
        A = np.dot(X.T, X) 
    A_inv = np.linalg.inv(A)
    t_out=np.dot(X.T,t.reshape(-1,1))
    w = np.dot(A_inv,t_out)
    
    f_M = np.dot(X, w) #prediction

    error = calcError(f_M, M, t, w)

    return f_M, error

#Calculate training/validation error
def calcError(f_M, M, t, w):
    error=0
    for i in range(np.size(f_M,0)):                 #summation
        error = error + (f_M[i,0]-t[i])**2          #error = (1/N)*(y-t)^2
    
    if (M==9):
        lambda1 = 1.53e-8
        error = error/np.size(f_M,0) + (lambda1/np.size(f_M,0))*(np.linalg.norm(w))**2      #Error with regularization: error/N + (lambda/N)*||w||^2
    else:
        error = error/np.size(f_M,0)    #(1/N)*error summation

    print ("Error: %f" %error)
    return error

# main for part 1
train_error = []
valid_error = []
avg_error = []
for M in range(10):
    print("M =", M)
    t_error, v_error, a_error = dataGen(M)
    train_error.append(t_error)
    valid_error.append(v_error)
    avg_error.append(a_error)

plt.plot(range(10), train_error, c='red', label='training error')
plt.plot(range(10), valid_error, c='blue', label='validation error')
plt.plot(range(10), avg_error, c='green', label='average error')
plt.legend()
plt.title('Training and Validation Error vs. M')
plt.xlabel('M')
plt.ylabel('Error')
plt.show()