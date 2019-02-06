# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize


####PART1####
def sigmoid(z):
    return 1./(1.+np.exp(-z))

def part1():
    test=np.array([1500,1,0,-1,-1500])
    sigmoidTest = sigmoid(test)
    for i,t in enumerate(test):
        print 'sigmoid('+str(t)+') = '+str(sigmoidTest[i])
########


####PART2####
def costFunction(X,y,theta):
    return computeCost(theta,X,y),computeGradient(X,y,theta)

def computeCost(theta,X,y):
    minv = 1./len(y)
    h = sigmoid(theta.dot(X.T))
    cost = minv * (- y.dot(np.log(h)) - (1.- y).dot(np.log(1.- h)))
    return cost

def computeGradient(X,y,theta):
    minv = 1./len(y)
    h = sigmoid(theta.dot(X.T))
    return minv * (h-y).dot(X)

def part2():
    data = np.loadtxt('data/ex2data1.txt',delimiter=',',unpack=True).T
    X = np.c_[np.ones((data.shape[0],1)) ,data[:,:-1]]
    y = data[:,-1]
    n = X.shape[1]
    theta = np.zeros(n)

    cost,gradient = costFunction(X,y,theta)
    print 'With theta = '+str(theta)+' --> \n\tcost = '+str(cost)+'\tgradient = '+str(gradient)
########


####PART3####
def findOptimalTheta(X,y,theta):
    result = optimize.fmin( computeCost, x0=theta, args=(X,y), maxiter=400, full_output=True )
    theta = result[0]
    cost = result[1]
    return theta,cost

def part3():
    data = np.loadtxt('data/ex2data1.txt',delimiter=',',unpack=True).T
    X = np.c_[np.ones((data.shape[0],1)) ,data[:,:-1]]
    y = data[:,-1]
    n = X.shape[1]
    theta = np.zeros(n)

    theta,cost = findOptimalTheta(X,y,theta)
    print 'With optimal theta = '+str(theta)+' --> cost = '+str(cost)

    plt.figure()
    plt.scatter((X[y==0,1]),(X[y==0,2]),marker='o',color='y',label='Not admitted')
    plt.scatter((X[y==1,1]),(X[y==1,2]),marker='x',color='b',label='Admitted')
    plt.plot( np.array([min(X[:,1]),max(X[:,1])]) , (-1./theta[2])*(theta[1]*np.array([min(X[:,1]),max(X[:,1])])+theta[0]) )
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.xlim([25, 115])
    plt.ylim([25, 115])
    plt.legend()
    plt.show()
########

####PART4####
def predict(X,y,test,theta):
    theta = findOptimalTheta(X,y,theta)[0]
    prob = sigmoid(test.dot(theta))
    return 1 if prob>=0.5 else 0,theta

def part4():
    data = np.loadtxt('data/ex2data1.txt',delimiter=',',unpack=True).T
    X = np.c_[np.ones((data.shape[0],1)) ,data[:,:-1]]
    y = data[:,-1]
    n = X.shape[1]
    theta = np.zeros(n)

    test=np.array([1.,45.,85.])
    print 'The prediction for '+str(test[1:])+' is '+str(predict(X,y,test,theta)[0])+' (probability = '+str(sigmoid(test.dot(predict(X,y,test,theta)[1])))+')'

########

if __name__ == '__main__':
    part1()
    part2()
    part3()
    part4()
