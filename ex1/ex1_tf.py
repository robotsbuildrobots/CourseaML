import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
import os

# Coursera ML ex1 in TensorFlow, using the TF Getting Started excercise and my python port of ex1 as a basis, 

def plotData():
	plt.figure(figsize=(10,10))
	plt.scatter(data[:,0], data[:,1])
	plt.show()
	## add labels

def plotTheta():
	x_axis = np.linspace(data[:,0].min(), data[:,0].max(), 100) #Returns evenly spaced numbers over specified interval, to make the theta line fit the scatter plot dimensions
	y_axis = final_theta[0] + (final_theta[1] * x_axis)
	plt.scatter(data[:,0], data[:,1])
	plt.plot(x_axis, y_axis)
	plt.show()
	## add labels etc

def plotCost():
	x_axis = np.arange(iters)
	y_axis = j_history
	plt.plot(x_axis, y_axis)
	plt.show()

def thetaShape(theta):
	print "Theta Shape:"
	print(theta.shape)

def featureNormalisation(data):
    mu = np.mean(data,axis=0)
    sigma = np.std(data,axis=0)
    return (data - mu)/sigma

def printTrainingData():
	print "======= Original Data ========"
	print(data)

def printNormalisedData():
	print "====== Normalised Data ======="
	print(data)

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

def printFinalValues():
	print "======== Cost History ========="
	print(j_history)
	print "======== Theta History ========="
	print(theta_history)
	print "======== Final Theta ========="
	print(final_theta)

#### Cleaning up data
data = np.loadtxt("ex1data1.txt", delimiter=",")
#printTrainingData()
#plotData()
data = featureNormalisation(data)
#printNormalisedData()

#### Initialising variables
x_temp = data[:,[0]] # slicing the x column from the data file into a temp column vector in order to make matrix x 
x = np.insert(x_temp, 0, values=1, axis=1) # create a new matrix starting with column vector x_temp, add a new column of 1s at index 0
y = data[:,[1]] # Note: square brackets around the index  means x is column vector!
iters = 50
alpha = 0.002
j_history = []
theta_history = []

#### All operations take place within a tensorflow session
with tf.Session() as sess:
	# Importing variables into the tensorflow session, and making sure to set them as the same type. Errors are thrown if this is not true.
	x = tf.constant(x.astype(np.float32))
	y = tf.constant(np.transpose([y]).astype(np.float32))
	theta = tf.Variable(tf.random_normal([2, 1], 1, 0.1))
	# Variables must then be explicitly initialised before they can be used in the computation
	tf.global_variables_initializer().run()
	#Now we set up the operations within the tensorflow session, using special tf functions
	hypothesis = tf.matmul(x, theta)
	error = tf.sub(hypothesis, y)
	cost = tf.nn.l2_loss(error) # l2_loss calculates squared error
	computeTheta = tf.train.GradientDescentOptimizer(alpha).minimize(cost) #specific gradient descent function. Give it the training rate and thing to minimise
	# Everything is set up. Now we set up the training loop and call computeTheta for the number of iterations we want.
	for i in range(iters):
		# Repeatedly run the gradient descent operation, updating the TensorFlow variable.
		sess.run(computeTheta)
		#keep a history of theta and the cost so we can see them converge
		theta_history.append(theta.eval())
		j_history.append(cost.eval())
	# Training is done, get the final values for the charts
	final_theta = theta.eval()
	hypothesis = hypothesis.eval()


#printFinalValues()
plotTheta()
plotCost()