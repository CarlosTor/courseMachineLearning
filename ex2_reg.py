# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize


####PART1####
def sigmoid(z):
    return 1./(1.+np.exp(-z))

def part1():
    data = np.loadtxt('data/ex2data2.txt',delimiter=',',unpack=True).T
    X = np.c_[np.ones((data.shape[0],1)) ,data[:,:-1]]
    y = data[:,-1]
    n = X.shape[1]

    plt.figure()
    plt.scatter((X[y==0,1]),(X[y==0,2]),marker='o',color='y',label='y = 0')
    plt.scatter((X[y==1,1]),(X[y==1,2]),marker='x',color='b',label='y = 1')
    plt.xlabel('Microship Test 1')
    plt.ylabel('Microship Test 2')
    plt.legend()
    plt.show()
########


####PART2####
def mapFeature(X1,X2,degrees):
	newX = np.ones( (np.shape(X1)[0], 1) )
	for i in range(1, degrees+1):
		for j in range(0, i+1):
			term1 = X1**(i-j)
			term2 = X2**(j)
			term  = (term1*term2).reshape(np.shape(term1)[0],1)
			newX = np.hstack( (newX,term) )
	return newX

def part2():
    data = np.loadtxt('data/ex2data2.txt',delimiter=',',unpack=True).T
    X = np.c_[np.ones((data.shape[0],1)) ,data[:,:-1]]
    y = data[:,-1]
    n = X.shape[1]-1

    X=mapFeature(X[:,1],X[:,2],6)
    print 'New X with size '+str(X.shape)
########


####PART3####
def costFunctionReg(X,y,theta,lam):
    return computeCostReg(theta,X,y,lam),computeGradientReg(X,y,theta,lam)

def computeCostReg(theta,X,y,lam):
    minv = 1./len(y)
    h = sigmoid(theta.dot(X.T))
    cost = minv * (- y.dot(np.log(h)) - (1.- y).dot(np.log(1.- h))) + lam*minv/2. * np.sum(theta[1:]**2)
    return cost

def computeGradientReg(X,y,theta,lam):
    minv = 1./len(y)
    thetaReg = np.hstack([0,theta[1:]])
    print thetaReg.shape
    h = sigmoid(theta.dot(X.T))
    return minv * ((h-y).dot(X) + lam*thetaReg)

def findOptimalThetaReg(X,y,theta,lam):
    result = optimize.fmin( computeCostReg, x0=theta, args=(X,y,lam), maxiter=100000, full_output=True )
    theta = result[0]
    cost = result[1]
    return theta,cost

def part3():
    data = np.loadtxt('data/ex2data2.txt',delimiter=',',unpack=True).T
    X = np.c_[np.ones((data.shape[0],1)) ,data[:,:-1]]
    X = mapFeature(X[:,1],X[:,2],6)
    y = data[:,-1]
    n = X.shape[1]
    theta = np.zeros(n)
    lam = 0.01

    cost,gradient = costFunctionReg(X,y,theta,lam)
    print 'With initial theta = '+str(theta)+' --> cost = '+str(cost)

    theta,cost = findOptimalThetaReg(X,y,theta,lam)
    print 'With optimal theta = '+str(theta)+' --> cost = '+str(cost)
########


####PART4####
def part4():
    data = np.loadtxt('data/ex2data2.txt',delimiter=',',unpack=True).T
    X = np.c_[np.ones((data.shape[0],1)) ,data[:,:-1]]
    newX = mapFeature(X[:,1],X[:,2],6)
    y = data[:,-1]
    n = newX.shape[1]-1
    lambdas = [0.01,0.1,1.,10]

    for lam in lambdas:
        theta = np.zeros(n+1)
        theta,cost = findOptimalThetaReg(newX,y,theta,lam)
    	plt.text( 0.15, 1.4, 'Lam %.1f' % lam )
        plt.figure()
        plt.scatter((X[y==0,1]),(X[y==0,2]),marker='o',color='y',label='y = 0')
        plt.scatter((X[y==1,1]),(X[y==1,2]),marker='x',color='b',label='y = 1')
        u = np.linspace( -1, 1.5, 50 )
        v = np.linspace( -1, 1.5, 50 )
        z = np.zeros( (len(u), len(v)) )

        for i in range(0, len(u)):
            for j in range(0, len(v)):
                mapped = mapFeature(np.array([u[i]]), np.array([v[j]]), 6)
                z[i,j] = mapped.dot( theta )
        z = z.transpose()

        u, v = np.meshgrid( u, v )
        plt.contour( u, v, z, [0.0, 0.0], label='Decision Boundary' )

        plt.xlabel('Microship Test 1')
        plt.ylabel('Microship Test 2')
        plt.legend()
        plt.show()
########

if __name__ == '__main__':
    part1()
    part2()
    part3()
    part4()
