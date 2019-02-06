# -*- coding: utf-8 -*-

import numpy as np
import math as m
import matplotlib.pyplot as plt
from random import randint


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
    minv = 1./len(y)
    h = sigmoid(theta.dot(X.T))
    cost = minv * (- y.dot(np.log(h)) - (1.- y).dot(np.log(1.- h)))
    gradient = minv * (h-y).dot(X)
    return cost, gradient

def part2():
    data = np.loadtxt('data/ex2data1.txt',delimiter=',',unpack=True).T
    X = np.c_[np.ones((data.shape[0],1)) ,data[:,:-1]]
    y = data[:,-1]
    n = X.shape[1]
    theta = np.zeros(n)

    cost,gradient = costFunction(X,y,theta)
    print 'With theta = '+str(theta)+' --> \n\tcost = '+str(cost)+'\tgradient = '+str(gradient)
########


if __name__ == '__main__':
    part1()
    part2()
