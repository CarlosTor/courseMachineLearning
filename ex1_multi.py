# -*- coding: utf-8 -*-

import numpy as np
import math as m
import matplotlib.pyplot as plt

####PART1####
def part1():
    A = np.eye(5)
    print 'Result of part 1 is:\n'+str(A)
#########


####PART2.1####
def part2_1():
    X,y = np.loadtxt('data/ex1data1.txt',delimiter=',',unpack=True)
    plt.figure()
    plt.plot(X,y,'x',color='r')
    plt.ylabel('Profit in $10,000s');
    plt.xlabel('Population of City in 10,000s');
    plt.show()
#########

####PART2.2####
def hipothesis(X,theta):
    Xs = np.ones((X.shape[0],X.shape[1]+1))
    Xs[:,1:] = X
    return np.sum(theta.T*Xs,axis=-1)

def costFunctionMulti(X,y,theta):
    m = len(y)
    cost = 1.0/(2*m) * np.sum((hipothesis(X,theta)-y)**2)
    return cost

def gradientDescentMulti(X,y,theta,alpha,iterations):
    m = len(y)
    Xs = np.ones((X.shape[0],X.shape[1]+1))
    Xs[:,1:] = X
    for i in xrange(iterations):
        theta = theta - alpha/m * np.sum( (hipothesis(X,theta)-y).dot(Xs), axis=-1)
    return theta


####PART3.1####
def featureNormalize(X):
    mu = np.mean(X,axis=-1)
    sigma = np.std(X,axis=-1)
    return (X.T-mu)/sigma,mu,sigma

def part3_1():
    data = np.loadtxt('data/ex1data2.txt',delimiter=',',unpack=True)
    X = data[:-1,:]
    y = data[-1,:]
    X,muX,sigmaX = featureNormalize(X)
    print 'After normalising X:  max = '+str(np.max(X))+' ; min = '+str(np.min(X))
########


####PART3.2####
def part3_2():
    data = np.loadtxt('data/ex1data2.txt',delimiter=',',unpack=True)
    X = data[:-1,:]
    y = data[-1,:]
    X,muX,sigmaX = featureNormalize(X)
    n = X.shape[1]
    theta = np.zeros(n+1)
    alpha = 0.01
    iterations = 100000

    print 'Initializing gradient descent with:\n\talpha = '+str(alpha)+'\n\titerations = '+str(iterations)+'\n\ttheta = '+str(theta)+' (cost function = '+str(costFunctionMulti(X,y,theta))+')'
    theta=gradientDescentMulti(X,y,theta,alpha,iterations)
    print 'Result: \n\ttheta = '+str(theta)+' (cost function = '+str(costFunctionMulti(X,y,theta))+')'
########

if __name__ == '__main__':
    part3_1()
    part3_2()
