#import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
import math
import os


def plotData():
	# x1 = values where data1[2] is 1
	# x2 = values where data1[2] is 0

	plt.figure(figsize=(10,10))
	plt.scatter(data1[:,0], data1[:,1], c=data1[:,2])
	plt.show()
	## add labels

def plotTheta():
	x = np.linspace(data[:,0].min(), data[:,0].max(), 100) #Returns evenly spaced numbers over the range covered by the data, to make the theta line fit the scatter plot dimensions
	y = theta[0, 0] + (theta[0, 1] * x)
	plt.plot(x, y)
	plt.show()
	## add labels etc

def plotCost():
	x = np.arange(1500)
	y = j_history
	plt.plot(x, y)
	plt.show()

def featureNormalisation(data):
    mu = np.mean(data,axis=0)
    sigma = np.std(data,axis=0)
    return (data - mu)/sigma

def printTrainingData():
	print "======= Original Data ========"
	print(data1)


def debugPrintParams():
	print "===== initialised values ====="
	print "----------- data -------------"
	print "Number of values, m: %d" % (m)
	print "******** Matrix, x ***********"
	print(x)
	print "****** Column Vector, y ******"
	print(y)

	print "-------- parameters  ---------"
	print "Number of iterations: %d" % (iters)
	print "Step Value, Alpha: %d" % (alpha)
	print "Initial Theta values:" 
	print (theta) #TODO: make nicer
	print "------------------------------"

def debugGDShape(theta, hypothesis, difference):
	print "Theta Shape:"
	print(theta.shape)
	print "Hypothesis:"
	print(hypothesis.shape)
	print "Difference:"
	print(difference.shape)

def debugGDIters(theta):
	print "Theta Shape:"
	print(theta.shape)
	print "Iteration: ", i
	print(theta) 

def sigmoid(x):
	s = 1 / (1 + np.exp(-x))
	return s

def costLogisticRegression(x, y, theta):
	m = len(y)
	#z = np.multiply(x, theta.T) #TODO - not multiplying matrix element-wise?! z is (100,1), should be (100,2)
	#z = np.dot(x, theta.T)
	z = x * theta.T #TODO - not multiplying matrix element-wise?! z is (100,1), should be (100,2)
	print('z:')
	print(z)
	hX = sigmoid(z)
	print('hypothesis:')
	print(hX)
	thing1 = np.sum(-y * np.log(hX.T))
	print('thing1')
	print(thing1)
	thing2 = np.sum((1 - y) * np.log(1 - hX.T))
	print('thing2')
	print(thing2)
	cost = (1/m) * thing1 - thing2
	print('cost')
	print(cost)
	return cost

def gradientLogisticRegression(x, y, theta, alpha, iters):
	####
	#hypothesis = sigmoid(X * theta);
	#J = (1/m) * sum(-y .* log(hypothesis) - (1 - y) .* log(1 - hypothesis));
	#grad = (1/m) * X' * (hypothesis - y);
	####

	m = len(y) # number of training examples
	j_history = np.zeros(iters)
	temp_theta = np.matrix(np.zeros(theta.shape))
	print "---running gradient descent---"	

	for i in range(iters): 
		difference = sigmoid(x * theta.T) - y
		#debugGDShape(theta, hypothesis, difference)
		temp_theta[:,[0]] = theta[:,[0]] - (
			(1 / m) * np.sum(np.multiply(x[:,[0]], difference)))
		temp_theta[:,[1]] = theta[:,[1]] - (
			(1 / m) * np.sum(np.multiply(x[:,[1]], difference)))
		theta = temp_theta
		j_history[i] = costLogisticRegression(x, y, theta)
		#debugGDIters(theta)
		#input('Press enter to continue: ')
	return theta, j_history



##########################################
# ==================== Part 1: Plotting ====================
#  We start the exercise by first plotting the data to understand the 
#  the problem we are working with.
##########################################

data1 = np.loadtxt("ex2data1.txt", delimiter=",")
data2 = np.loadtxt("ex2data2.txt", delimiter=",")
printTrainingData()
#plotData()

print(data1)
data = data1
#normalise



##########################################
# ============ Part 2: Compute Cost and Gradient ============
# In this part of the exercise, you will implement the cost and gradient
# for logistic regression. 
##########################################




x1 = data[:,[1]]
x2 = data[:,[2]] # slicing the x column from the data file into a temp column vector in order to make matrix x 
y = data[:,[0]] # Note: square brackets around the index  means x is column vector!
m = len(y)


x1 = featureNormalisation(x1)
print('======Normalised Data=======')
print(x1)
x = np.append(x1, x2, 1) # create a new matrix starting with column vector x_temp, add a new column of 1s at index 0

print('======reconstructed x=======')
print(x)
y = featureNormalisation(y)

iters = 1500
alpha = 0.001

x = np.matrix(x)
y = np.matrix(y)
theta = np.matrix(np.ones((1,2), int))
debugPrintParams()

s1 = sigmoid(1)
print('sigmoid (1):')
print(s1)
s0 = sigmoid(0)
print('sigmoid (0):')
print(s0)
print('sigmoid (x):')
sX = sigmoid(x)
print(sX)


cost = costLogisticRegression(x, y, theta)

print "Initial Computed Cost:" 
print(cost)
print(cost.shape)

#theta, j_history = gradientLogisticRegression(x, y, theta, alpha, iters)
'''
print "================================="
print "Theta, found by gradient descent:" 
print(theta)
print(theta.shape)
print "================================="
print "Cost:"
print(j_history)
print(j_history.shape)
'''
#plotTheta()

#plotCost()

###########################################
#  In this exercise, you will use a built-in function (fminunc) to find the
#  optimal parameters theta.
###########################################



###########################################
# ============== Part 4: Predict and Accuracies ==============
#  After learning the parameters, you'll like to use it to predict the outcomes
#  on unseen data. In this part, you will use the logistic regression model
# to predict the probability that a student with score 45 on exam 1 and 
#  score 85 on exam 2 will be admitted.
#
#  Furthermore, you will compute the training and test set accuracies of 
#  our model.
#
#  Your task is to complete the code in predict.m
#  Predict probability for a student with score 45 on exam 1 
#  and score 85 on exam 2 
###########################################



