# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize,io,misc


####PART1.1####
def part1_1():
    data = io.loadmat('data/ex3data1.mat')
    X,y = data['X'],data['y']
#########


####PART1.2####
def displayData(X):
    width = int(X.shape[1]**0.5)
    rows,cols = 15,15
    picture = np.zeros((width*rows,width*cols))

    rand_indices = np.random.permutation( 5000 )[:rows*cols]

    count = 0
    for y in range(0, rows):
        for x in range(0, cols):
            start_x = x*width
            start_y = y*width
            picture[x*width:x*width+width,y*width:y*width+width] = X[rand_indices[count]].reshape(width,width).T
            count += 1

    img = misc.toimage( picture )
    figure = plt.figure()
    axes = figure.add_subplot(111)
    plt.axis('off')
    axes.imshow( img )
    plt.show()

def part1_2():
    data = io.loadmat('data/ex3data1.mat')
    X,y = data['X'],data['y']
    displayData(X)
########


####PART1.3####
def sigmoid(z):
    return 1./(1.+np.exp(-z))

def costFunctionReg(X,y,theta,lam):
    return computeCostReg(theta,X,y,lam),computeGradientReg(theta,X,y,lam)

def computeCostReg(theta,X,y,lam):
    minv = 1./len(y)
    h = sigmoid(theta.dot(X.T))
    cost = minv * (- y.dot(np.log(h)) - (1.- y).dot(np.log(1.- h))) + lam*minv/2. * np.sum(theta[1:]**2)
    return cost

def computeGradientReg(theta,X,y,lam):
    minv = 1./len(y)
    thetaReg = np.hstack([0,theta[1:]])
    h = sigmoid(theta.dot(X.T))
    return minv * ((h-y).dot(X) + lam*thetaReg)

def findOptimalThetaReg(X,y,theta,lam):
    result = optimize.fmin_cg( computeCostReg, fprime=computeGradientReg, x0=theta, args=(X,y,lam), maxiter=100, disp=False, full_output=True )
    theta = result[0]
    cost = result[1]
    return theta,cost

def oneVsAllLogisticRegression(X,y,lam):
    numClass = len(np.unique(y))
    theta = np.zeros((X.shape[1],numClass))
    initCost,finalCost = 0,0
    for c in xrange(numClass):
        tmpY = 1*(y==c+1).reshape(-1)
        initCost += computeCostReg(theta[:,c],X,tmpY,lam)
        theta[:,c],tmpCost =findOptimalThetaReg(X,tmpY,theta[:,c],lam)
        finalCost += tmpCost
    return theta,finalCost,initCost


def part1_3():
    data = io.loadmat('data/ex3data1.mat')
    X,y = data['X'],data['y']
    X = np.c_[np.ones((X.shape[0],1)) ,X]
    lam = 0.6

    theta,finalCost,initCost = oneVsAllLogisticRegression(X,y,lam)
    print 'With initial theta --> cost = '+str(initCost)
    print 'With optimal theta --> cost = '+str(finalCost)
########


####PART1.4####
def predictOneVsAllLogisticRegression(theta,X,y):
    m = X.shape[0]
    correct = 0.0
    for ex in xrange(m):
        prediction = np.argmax(theta.T.dot(X[ex,:]))+1
        result = y[ex]
        if prediction == result:
            correct += 1.0
    rate = correct / (m+0.0)
    return rate

def part1_4():
    data = io.loadmat('data/ex3data1.mat')
    X,y = data['X'],data['y']
    X = np.c_[np.ones((X.shape[0],1)) ,X]
    lam = 0.6

    theta,finalCost,initCost = oneVsAllLogisticRegression(X,y,lam)
    correctRate = predictOneVsAllLogisticRegression(theta,X,y)
    print 'The accuracy of training parameters are '+str(correctRate*100)+' %'
########



if __name__ == '__main__':
    part1_1()
    part1_2()
    part1_3()
    part1_4()
