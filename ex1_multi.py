# -*- coding: utf-8 -*-

import numpy as np
import math as m
import matplotlib.pyplot as plt
from random import randint



####PART3.1####
def loadData(normalize):
    data = np.loadtxt('data/ex1data2.txt',delimiter=',',unpack=True).T
    X = data[:,:-1]
    y = data[:,-1]
    if normalize:
        X,muX,sigmaX = featureNormalize(X)
    else:
        muX,sigmaX = np.mean(X,axis=0),np.std(X,axis=0)
    Xs = np.ones((X.shape[0],X.shape[1]+1))
    Xs[:,1:] = X
    return Xs,X,y,muX,sigmaX

def featureNormalize(data):
    mu = np.mean(data,axis=0)
    sigma = np.std(data,axis=0)
    return (data-mu)/sigma,mu,sigma

def part3_1():
    Xs,X,y,muX,sigmaX = loadData(normalize=True)
    print 'After normalising X:  max = '+str(np.max(X))+' ; min = '+str(np.min(X))
########


####PART3.2####
def randomColors(length):
    colors=list()
    for i in range(length):
        colors.append('#%06X' % randint(0, 0xFFFFFF))
    return colors

def costFunctionMulti(X,y,theta):
    m = len(y)
    cost = 1.0/(2*m) * np.sum((theta.dot(X.T)-y)**2)
    return cost

def gradientDescentMulti(X,y,theta,alpha,iterations):
    m = len(y)
    costIteration=list()
    alpham = alpha/m
    for i in xrange(iterations):
        theta = theta - alpham * (theta.dot(X.T)-y).dot(X)
        costIteration.append(costFunctionMulti(X,y,theta))
    return theta,costIteration

def part3_2():
    Xs,X,y,muX,sigmaX = loadData(normalize=True)
    n = X.shape[1]
    alphas = [0.01,0.03,0.1,0.3,1.]

    colors=randomColors(len(alphas))
    iterations = 400
    for i,alpha in enumerate(alphas):
        theta = np.zeros(n+1)
        print '\nInitializing gradient descent with:\n\talpha = '+str(alpha)+'\n\titerations = '+str(iterations)+'\n\ttheta = '+str(theta)+' (cost function = '+str(costFunctionMulti(Xs,y,theta))+')'
        theta,costIteration=gradientDescentMulti(Xs,y,theta,alpha,iterations)
        print 'Result of gradient decent: \n\ttheta = '+str(theta)+' (cost function = '+str(costFunctionMulti(Xs,y,theta))+')'
        plt.figure(1)
        plt.plot(costIteration,color=colors[i])
        plt.xlabel('Number of iterations')
        plt.ylabel('Cost function')
        plt.xlim([0, 50])
    plt.show()
########


####PART3.3####
def normalEquation(X,y):
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

def part3_3():
    Xs,X,y,muX,sigmaX = loadData(normalize=False)
    n = X.shape[1]

    theta = normalEquation(Xs,y)
    print 'Result of normal equation: \n\ttheta = '+str(theta)+' (cost function = '+str(costFunctionMulti(Xs,y,theta))+')'

    Xs,X,y,muX,sigmaX = loadData(normalize=True)
    input = np.array((1.,1650.,3.))
    normInput = input.copy()
    normInput[1:] = (input[1:]-muX)/sigmaX
    print 'Test gradient descent:\t'+ str(gradientDescentMulti(Xs,y,np.zeros(n+1),1.,400)[0].dot(normInput))
    print 'Test normal equation:\t'+ str(theta.dot(input))
########

if __name__ == '__main__':
    part3_1()
    part3_2()
    part3_3()
