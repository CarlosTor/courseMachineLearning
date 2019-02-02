# -*- coding: utf-8 -*-

import numpy as np
import math as m
import matplotlib.pyplot as plt

####PART1####
def part1():
    A = np.eye(5)
    print 'Result of part 1 is: '+str(A)
#########


####PART2.1####
def part2_1():
    X,Y = np.loadtxt('data/ex1data1.txt',delimiter=',',unpack=True)
    plt.figure()
    plt.plot(X,Y,'x')
    plt.show()
#########


if __name__ == '__main__':
    part1()
    part2_1()
