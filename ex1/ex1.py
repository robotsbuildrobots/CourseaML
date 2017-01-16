import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
import os

def simpleMatrix():
	n = np.zeros((5,5), int)
	print(n)

def diagonalMatrix():
	k = np.fill_diagonal(n,1) #TODO
	print(n)

def plotData():
	plt.figure(figsize=(10,10))
	plt.scatter(data[:,0], data[:,1])
	plt.show()
	## add labels

def plotTheta():
	x = np.linspace(data[:,0].min(), data[:,0].max(), 100) #Returns evenly spaced numbers over the range covered by the data, to make the theta line fit the scatter plot dimensions
	y = theta[0, 0] + (theta[0, 1] * x)

	plt.scatter(data[:,0], data[:,1])
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
	print(data)

def printNormalisedData():
	print "====== Normalised Data ======="
	print(norm_data)

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

def computeCost(x, y, theta):

	m = len(y)
	hypothesis = x * theta.T
	sq_error = np.power((hypothesis - y), 2)
	return np.sum(sq_error) / (2*m) #TODO

def gradientDescent(x, y, theta, alpha, iters):
	m = len(y) # number of training examples
	j_history = np.zeros(iters)
	temp_theta = np.matrix(np.zeros(theta.shape))
	print "---running gradient descent---"	

	for i in range(iters): 
		difference = (x * theta.T) - y
		#debugGDShape(theta, hypothesis, difference)
		temp_theta[:,[0]] = theta[:,[0]] - ((alpha / m) * np.sum(np.multiply(x[:,[0]], difference)))
		temp_theta[:,[1]] = theta[:,[1]] - ((alpha / m) * np.sum(np.multiply(x[:,[1]], difference)))
		theta = temp_theta
		j_history[i] = computeCost(x, y, theta)
		#debugGDIters(theta)
		#input('Press enter to continue: ')
	return theta, j_history

	#Issues:
	#Same values for theta[0], theta[1]. Need to compute them separately?


##########################################
# Part 1: warmup excercise
# 1. make a 5x5 matrix with zeroes
# 2. print it 
# 3. make a 5x5 matrix with diagonal ones
##########################################

# simpleMatrix()
# diagonalMatrix()

##########################################
# Part 2: plot the data from ex1data1.txt
## get the excercise data into an array
##########################################

data = np.loadtxt("ex1data1.txt", delimiter=",")
#printTrainingData()
#plotData()

###########################################
# Part 5: Feature normalisation: % Instructions: First, for each feature dimension, compute the mean of the feature and subtract it from the dataset, storing the mean value in mu. Next, compute the standard deviation of each feature and divide each feature by it's standard deviation, storing the standard deviation in sigma. 
# Note that X is a matrix where each column is a feature and each row is an example. You need to perform the normalization separately for each feature.  Hint: You might find the 'mean' and 'std' functions useful.
###########################################

#norm_data = featureNormalisation(data)
#printNormalisedData()

###########################################
# Part 3: 
# Objective of linear regression is to minimise the cost function.
# hypothesis: h(x) = theta_0 + (theta_1*x)
# 3.1 Compute and display the initial cost
###########################################

x_temp = data[:,[0]] # slicing the x column from the data file into a temp column vector in order to make matrix x 
y = data[:,[1]] # Note: square brackets around the index  means x is column vector!
m = len(y)
x = np.insert(x_temp, 0, values=1, axis=1) # create a new matrix starting with column vector x_temp, add a new column of 1s at index 0

iters = 1500
alpha = 0.001

x = np.matrix(x)
y = np.matrix(y)
theta = np.matrix(np.array([0,0]))

debugPrintParams()

j = computeCost(x, y, theta)

print "Initial Computed Cost:" 
print(j)
print(j.shape)

##########################################
# 3.2 Run gradient descent, print Theta Instructions: Perform a single gradient step on the parameter vector theta. 
# Hint: While debugging, it can be useful to print out the values of the cost function (computeCost) and gradient here.
##########################################

theta, j_history = gradientDescent(x, y, theta, alpha, iters)

print "================================="
print "Theta, found by gradient descent:" 
print(theta)
print(theta.shape)
print "================================="
print "Cost:"
print(j_history)
print(j_history.shape)

##########################################
# 3.3 Plot the linear fit
##########################################

plotTheta()

plotCost()


##########################################
# 3.4 predict values fro population sizes of 35000 and 70000
##########################################





##########################################
# Part 4: Visualise J(theta_0, theta_1) with a surface plot
##########################################





  
