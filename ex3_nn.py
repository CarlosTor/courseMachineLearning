# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize,io,misc


####PART2.1####
def part2_1():
	data = io.loadmat('data/ex3data1.mat')
	X,y = data['X'],data['y']
	data = io.loadmat('data/ex3weights.mat')
	Theta1 = data['Theta1']
	Theta2 = data['Theta2']
	print 'Our already trained parameters Theta 1 ('+str(Theta1.shape)+') and Theta 2 ('+str(Theta2.shape)+') are loaded'
#########


####PART2.2####
def sigmoid(z):
    return 1./(1.+np.exp(-z))

def predictNeuralNetwork(X,y,*Thetas):
	m = X.shape[0]
	correct = 0.0
	for ex in xrange(m):
		A = list()
		A.append(X[ex,:])
		for i in xrange(len(Thetas)):
			a = np.zeros(Thetas[i].shape)
			z = Thetas[i].dot(np.hstack((1,A[-1])))
			A.append(sigmoid(z.T))
		prediction = np.argmax(A[-1])+1
		result = y[ex]
		if prediction == result:
			correct += 1.0
	rate = correct / (m+0.0)
	return rate

def part2_2():
	data = io.loadmat('data/ex3data1.mat')
	X,y = data['X'],data['y']
	data = io.loadmat('data/ex3weights.mat')
	Theta1 = data['Theta1']
	Theta2 = data['Theta2']
	correctRate = predictNeuralNetwork(X,y,Theta1,Theta2)
	print 'The accuracy of the trained Neural Network is '+str(correctRate*100)+' %'
########


if __name__ == '__main__':
    part2_1()
    part2_2()
