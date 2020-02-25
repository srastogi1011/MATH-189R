"""
Start file for hw4pr2 part (b) of Big Data Summer 2017

The file is seperated into two parts:
	1) the helper functions
	2) the main driver.

The helper functions are all functions necessary to finish the problem.
The main driver will use the helper functions you finished to report and print
out the results you need for the problem.

First, please COMMENT OUT any steps other than step 0 in main driver before
you finish the corresponding functions for that step. Otherwise, you won't be
able to run the program because of errors.

After finishing the helper functions for each step, you can uncomment
the code in main driver to check the result.

Note:
1. Please read the instructions and hints carefully, and use the name of the
variables we provided, otherwise, the function may not work.

2. Remember to comment out the TODO comment after you finish each part.
"""


#########################################
#			 Helper Functions	    	#
#########################################

import p2_data as data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import time



def NLL(X, y, W, reg=0.0):
	"""	This function takes in four arguments:
			1) X, the data matrix with dimension m x (n + 1)
			2) y, the label of the data with dimension m x 1
			3) W, a weight matrix
			4) reg, the parameter for regularization

		This function calculates and returns negative log likelihood for
		softmax regression.

		HINT:
			1) Recall the negative log likelihood function for softmax
			   regression and
			2) Use a.sum() to find the summation of all entries in a numpy
			   array a
			3) When perform operations vertically across rows, we use axis=0.
			   When perform operations horizontally across columns, we use
			   axis=1.
			4) Use np.exp and np.log to calculate the exp and log of
			   each entry of the input array

		NOTE: Please use the variable given, NLL.
	"""
	##LOOKED AT SOLUTION
	"*** YOUR CODE HERE ***"
	mu = X @ W  # m x k
	exp_mu = np.exp(mu)
	prob = exp_mu / exp_mu.sum(axis=1).reshape(-1, 1)
	groundTruth = y * np.log(prob)
	NLL = -groundTruth.sum(axis=1).sum() + reg * np.diag(W.T @ W).sum()
	"*** END YOUR CODE HERE ***"
	return NLL



def grad_softmax(X, y, W, reg=0.0):
	"""	This function takes in four arguments:
			1) X, the data matrix with dimension m x (n + 1)
			2) y, the label of the data with dimension m x 1
			3) W, a weight matrix
			4) reg, the parameter for regularization

		This function calculates and returns the gradient of W for softmax
		regression.

		HINT:
			1) Recall the log likelihood function for softmax regression and
			   get the gradient with respect to the weight matrix, W
			2) Remember to apply the regularization

		NOTE: Please use the variable given for the gradient, grad.
	"""

	"*** YOUR CODE HERE ***"
	exp_mu = np.exp(X @ W)
	val = exp_mu / exp_mu.sum(axis=1).reshape(-1, 1) #LOOKED AT SOLUTION
	grad = X.T @ (val - y) + reg * W

	"*** END YOUR CODE HERE ***"
	return grad



def predict(X, W):
	"""	This function takes in two arguments:
			1) X, the data matrix with dimension m x (n + 1)
			2) W, a weight matrix

		This function returns the predicted labels y_pred with
		dimension m x 1

		HINT:
			1) Firstly obtain the probablity matrix according to the softmax
			   equation
			2) Use np.argmax to get the predicted label for each image

		NOTE: Please use the variable given, y_pred.
	"""

	# Weight given
	"*** YOUR CODE HERE ***"
	exp_mu = np.exp(X @ W)
	val = exp_mu / exp_mu.sum(axis=1).reshape(-1, 1)
	y_pred = np.argmax(val, axis=1).reshape(-1, 1)
	"*** END YOUR CODE HERE ***"
	return y_pred



def get_accuracy(y_pred, y):
	"""	This function takes in two arguments:
			1) y_pred, the predicted label of data with dimension m x 1
			2) y, the true label of data with dimension m x 1

		This function calculates and returns the accuracy of the prediction

		NOTE: You DO NOT need to change this function
	"""
	diff = (y_pred == y).astype(int)
	accu = 1. * diff.sum() / len(y)
	return accu




def grad_descent(X, y, reg=0.0, lr=1e-5, eps=1e-6, max_iter=500, print_freq=20):
	"""	This function takes in seven arguments:
			1) X, the data with dimension m x (n + 1)
			2) y, the label of data with dimension m x 1
			3) reg, the parameter for regularization
			4) lr, the learning rate
			5) eps, the threshold of the norm for the gradients
			6) max_iter, the maximum number of iterations
			7) print_freq, the frequency of printing the report

		This function returns W, the optimal weight by gradient descent,
		and nll_list, the corresponding learning objectives.
	"""
	# get the shape of the data, and initialize nll_list
	m, n = X.shape
	k = y.shape[1]
	nll_list = []

	# initialize the weight and its gradient
	W = np.zeros((n, k))
	W_grad = np.ones((n, k))


	print('\n==> Running gradient descent...')

	# Start iteration for gradient descent
	iter_num = 0
	t_start = time.time()


	# HINT: Run the gradient descent algorithm followed steps below
	#	1) Calculate the negative log likelihood at each iteration use function
	#	   NLL defined above
	#	2) Use np.isnan to test element-wise for NaN in obtained nll. If isnan,
	#	   break out of the while loop
	#	3) Otherwise, append the nll to the nll_list
	#	4) Calculate the gradient for W using grad_softmax defined above
	#	5) Upgrade W
	#	6) Keep iterating while the number of iterations is less than the
	#	   maximum and the gradient is larger than the threshold

	# NOTE: When calculating negative log likelihood at each iteration, please
	#		use variable name nll to store the value. Otherwise, there might be
	# 		error when you run the code.

	while iter_num < max_iter and np.linalg.norm(W_grad) > eps:

		"*** YOUR CODE HERE ***"
		nll = NLL(X, y, W, reg)
		if np.isnan(nll):
			break
		nll_list.append(nll)
		W_grad = grad_softmax(X, y, W, reg)
		W -= lr * W_grad

		"*** END YOUR CODE HERE ***"

		# Print statements for debugging
		if (iter_num + 1) % print_freq == 0:
			print('-- Iteration {} - negative log likelihood {: 4.4f}'.format(\
					iter_num + 1, nll))

		# Goes to the next iteration
		iter_num += 1


	# benchmark
	t_end = time.time()
	print('-- Time elapsed for running gradient descent: {t:2.2f} seconds'.format(\
			t=t_end - t_start))

	return W, nll_list



def accuracy_vs_lambda(X_train, y_train_OH, X_test, y_test, lambda_list):
	"""	This function takes in five arguments:
			1) X_train, the training data with dimension m x (n + 1)
			2) y_train_OH, the label of training data with dimension m x 1
			3) X_test, the validation data with dimension m x (n + 1)
			4) y_test, the label of validation data with dimension m x 1
			5) lambda_list, a list of different regularization paramters that
							we want to test

		This function generates a plot of accuracy of prediction vs lambda and
		returns the regularization parameter that maximizes the accuracy,
		reg_opt.

		HINT: generate the list of accuracy following the steps below:
			1) Run gradient descent with each parameter to obtain the optimal
			   weight
			2) Predicted the label using the weights
			3) Use get_accuracy function provided to calculate the accuracy
	"""
	# initialize the list of accuracy
	accu_list = []


	for reg in lambda_list:

		"*** YOUR CODE HERE ***"
		for l in lambda_list:
			W_opt, nll_list = grad_descent(X_train, y_train_OH, l)
			y_pred = predict(X_test, W_opt)
			accu_list.append(get_accuracy(y_pred, y_test))

		"*** END YOUR CODE HERE ***"

		print('-- Accuracy is {:2.4f} for lambda = {:2.2f}'.format(accuracy, reg))


	# Plot accuracy vs lambda
	print('==> Printing accuracy vs lambda...')
	plt.style.use('ggplot')
	plt.plot(lambda_list, accu_list)
	plt.title('Accuracy versus Lambda in Softmax Regression')
	plt.xlabel('Lambda')
	plt.ylabel('Accuracy')
	plt.savefig('hw4pr2b_lva.png', format = 'png')
	plt.close()
	print('==> Plotting completed.')


	# NOTE: use the variable given, reg_opt
	"*** YOUR CODE HERE ***"
	index = np.argmax(accu_list)
	reg_opt = lambda_list[index]
	"*** END YOUR CODE HERE ***"

	return reg_opt






###########################################
#	    	Main Driver Function       	  #
###########################################

# You should comment out the sections that
# you have not completed yet

if __name__ == '__main__':


	# =============STEP 0: LOADING DATA=================
	# NOTE: The data is loaded using the code in p2_data.py. Please make sure
	#		you read the code in that file and understand how it works.

	df_train = data.df_train
	df_test = data.df_test

	X_train = data.X_train
	y_train = data.y_train
	X_test = data.X_test
	y_test = data.y_test

	# stacking an array of ones
	X_train = np.hstack((np.ones_like(y_train), X_train))
	X_test = np.hstack((np.ones_like(y_test), X_test))

	# one hot encoder
	enc = OneHotEncoder()
	y_train_OH = enc.fit_transform(y_train.copy()).astype(int).toarray()
	y_test_OH = enc.fit_transform(y_test.copy()).astype(int).toarray()




	# =============STEP 1: Accuracy versus lambda=================
	# NOTE: Fill in the code in NLL, grad_softmax, predict and grad_descent.
	# 		Then, fill in predict and accuracy_vs_lambda

	print('\n\n==> Step 1: Finding optimal regularization parameter...')

	lambda_list = [0.01, 0.1, 0.5, 1.0, 10.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
	reg_opt = accuracy_vs_lambda(X_train, y_train_OH, X_test, y_test, lambda_list)

	print('\n-- Optimal regularization parameter is {:2.2f}'.format(reg_opt))





	# =============STEP 2: Convergence plot=================
	# NOTE: You DO NOT need to fill in any additional helper function for this
	# 		step to run. This step uses what you implemented for the previous
	#		step to plot.

	# run gradient descent to get the nll_list
	W_gd, nll_list_gd = grad_descent(X_train, y_train_OH, reg=reg_opt,\
	 	max_iter=1500, lr=2e-5, print_freq=100)

	print('\n==> Step 2: Plotting convergence plot...')

	# set up style for the plot
	plt.style.use('ggplot')

	# generate the convergence plot
	nll_gd_plot, = plt.plot(range(len(nll_list_gd)), nll_list_gd)
	plt.setp(nll_gd_plot, color = 'red')

	# add legend, title, etc and save the figure
	plt.title('Convergence Plot on Softmax Regression with $\lambda = {:2.2f}$'.format(reg_opt))
	plt.xlabel('Iteration')
	plt.ylabel('NLL')
	plt.savefig('hw4pr2b_convergence.png', format = 'png')
	plt.close()

	print('==> Plotting completed.')
