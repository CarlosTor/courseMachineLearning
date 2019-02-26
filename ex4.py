# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize,io,misc


####PART1.1####
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

def part1_1():
	data = io.loadmat('data/ex4data1.mat')
	X,y = data['X'],data['y']
	displayData(X)
#########


####PART1.2####
def part1_2():
	data = io.loadmat('data/ex4data1.mat')
	X,y = data['X'],data['y']
	data = io.loadmat('data/ex3weights.mat')
	Theta1 = data['Theta1']
	Theta2 = data['Theta2']
	print 'Our already trained parameters Theta 1 ('+str(Theta1.shape)+') and Theta 2 ('+str(Theta2.shape)+') are loaded'
########


####PART1.3####
def sigmoid(z):
    return 1./(1.+np.exp(-z))

def feedforward(X,*Thetas):
    A = list()
    A.append(X)
    for i in xrange(len(Thetas)):
        z = Thetas[i].dot(np.c_[np.ones((A[-1].shape[0],1)),A[-1]].T)
        A.append(sigmoid(z.T))
    return A

def nnCostFunction(X,y,A,lam,*Theta):
    minv = 1./X.shape[0]
    h = A[-1]
    newY = np.zeros(A[-1].shape)
    newY[(range(y.shape[0]),(y-1).astype(int).T)] = 1
    cost = (minv * (- newY * (np.log(h)) - (1.- newY) * (np.log(1.- h)))).sum()
    for i in range(len(Theta)):
        cost += lam*minv/2. * np.sum(Theta[i]**2)
    return cost

def part1_3():
    data = io.loadmat('data/ex4data1.mat')
    X,y = data['X'],data['y']
    data = io.loadmat('data/ex3weights.mat')
    Theta1 = data['Theta1']
    Theta2 = data['Theta2']
    A = feedforward(X,Theta1,Theta2)
    cost = nnCostFunction(X,y,A,0,Theta1,Theta2)
    print 'Without regularisation, the cost is '+str(cost)
########


####PART1.4####
def part1_4():
    data = io.loadmat('data/ex4data1.mat')
    X,y = data['X'],data['y']
    data = io.loadmat('data/ex3weights.mat')
    Theta1 = data['Theta1']
    Theta2 = data['Theta2']
    A = feedforward(X,Theta1,Theta2)
    lam = 1.
    cost = nnCostFunction(X,y,A,lam,Theta1,Theta2)
    print 'With regularisation (lambda='+str(lam)+'), the cost is '+str(cost)
########





if __name__ == '__main__':
    part1_1()
    part1_2()
    part1_3()
    part1_4()
