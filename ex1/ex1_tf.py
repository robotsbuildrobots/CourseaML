import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
import os

# using the tensorflow lession 2  that into the language used in Coursera ML ex1
# Coursera ML ex1, using Getting Started and my python port of ex1 as a basis, 

def plotData():
	plt.figure(figsize=(10,10))
	plt.scatter(data[:,0], data[:,1])
	plt.show()
	## add labels

def plotTheta():
	x = np.linspace(data[:,0].min(), data[:,0].max(), 100) #Returns evenly spaced numbers over specified interval, to make the theta line fit the scatter plot dimensions
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

def thetaShape(theta):
	print "Theta Shape:"
	print(theta.shape)

def printTrainingData():
	print "======= Original Data ========"
	print(data)


data = np.loadtxt("ex1data1.txt", delimiter=",")
#printTrainingData()
#plotData()

x_temp = data[:,[0]] # slicing the x column from the data file into a temp column vector in order to make matrix x 
y = data[:,[1]] # Note: square brackets around the index  means x is column vector!
m = len(y)
x = np.insert(x_temp, 0, values=1, axis=1) # create a new matrix starting with column vector x_temp, add a new column of 1s at index 0

iters = 500
alpha = 0.0005

initial_theta = np.matrix(np.array([2,1]))
thetaShape(initial_theta)

j_history = []
theta_history = []


# All operations take place within a tensorflow session
with tf.Session() as sess:
	# Importing variables into the tensorflow session, and making sure to set them as the same type. Errors are thrown if this is not true.
    input = tf.constant(x.astype(np.float32))
    y = tf.constant(y.astype(np.float32))
    theta = tf.Variable(tf.random_normal([2, 1], 1, 0.1))

    # Variables must then be explicitly initialised before they can be used in the computation
    tf.global_variables_initializer().run()

    #Now we set up the operations within the tensorflow session, using special tf functions
    hypothesis = tf.matmul(input, theta)
    error = tf.sub(hypothesis, y)
    cost = tf.nn.l2_loss(error) # l2_loss calculates squared error
    computeTheta = tf.train.GradientDescentOptimizer(alpha).minimize(cost) #specific gradient descent function. Give it the training rate


    # Everything is set up. Now we set up the training loop and call computeTheta for the number of iterations we want.
    for i in range(iters):
        # Repeatedly run the operations, updating the TensorFlow variable.
        sess.run(computeTheta)

        #keep a history of theta and the cost so we can see them converge
        theta_history.append(theta.eval())
        j_history.append(cost.eval())

    # Training is done, get the final values for the charts
    final_theta = theta.eval()

print(theta_history)
print(j_history)
print(final_theta)
plotTheta()
plotCost()