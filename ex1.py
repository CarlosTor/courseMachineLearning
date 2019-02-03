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
    Xs = np.array([np.ones(X.shape[0]),X])
    return np.sum(theta*Xs.T,axis=-1)

def costFunction(X,y,theta):
    m = len(y)
    cost = 1.0/(2*m) * np.sum((hipothesis(X,theta)-y)**2)
    return cost

def gradientDescent(X,y,theta,alpha,iterations):
    m = len(y)
    for i in xrange(iterations):
        theta = theta - alpha/m * np.sum( (hipothesis(X,theta)-y)*np.array([np.ones(len(y)),X]), axis=-1)
    return theta

def part2_2():
    X,y = np.loadtxt('data/ex1data1.txt',delimiter=',',unpack=True)
    theta = np.array([0,0])
    alpha = 0.01
    iterations = 1500

    print 'Initializing gradient descent with:\n\talpha = '+str(alpha)+'\n\titerations = '+str(iterations)+'\n\ttheta = '+str(theta)+' (cost function = '+str(costFunction(X,y,theta))+')'
    theta=gradientDescent(X,y,theta,alpha,iterations)
    print 'Result: \n\ttheta = '+str(theta)+' (cost function = '+str(costFunction(X,y,theta))+')'

    plt.figure()
    plt.plot(X,y,'x',color='r')
    plt.ylabel('Profit in $10,000s');
    plt.xlabel('Population of City in 10,000s');
    plt.plot(X, hipothesis(X,theta), 'b-')
    plt.show()
########

if __name__ == '__main__':
    part1()
    part2_1()
    part2_2()
