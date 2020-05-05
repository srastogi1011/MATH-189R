"""
Start file for hw4pr2 part(a) of Big Data Summer 2017

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
import time


#########################
#	    Step 1a`		#
#########################

def sigmoid(x):
	"""	This function takes in one argument:
			1) x, a numpy array

		This function applies the sigmoid / logistic function on each entry of
		the input array returns the new array.

		NOTE: You don't need to change this function.
	"""
	return 1. / (1. + np.exp(-x))



def grad_logreg(X, y, W, reg=0.0):
	"""	This function takes in four arguments:
			1) X, the data matrix with dimension m x (n + 1)
			2) y, the label of the data with dimension m x 1
			3) W, a weight matrix with bias
			4) reg, the parameter for regularization

		This function calculates and returns the gradient of W for logistic
		regression.

		HINT:
			1) Recall the log likelihood function for logistic regression and
			   get the gradient with respect to the weight matrix, W
			2) Remember to apply the l2 regularization
			3) You will need to use the sigmoid function above

		NOTE: Please use the variable given for the gradient, grad.
	"""
	##LOOKED AT SOLUTION
	"*** YOUR CODE HERE ***"
	grad = X.T @ (sigmoid(X @ W) - y) + reg * W

	"*** END YOUR CODE HERE ***"
	return grad




def NLL(X, y, W, reg=0.0):
	"""	This function takes in four arguments:
			1) X, the data matrix with dimension m x (n + 1)
			2) y, the label of the data with dimension m x 1
			3) W, a weight matrix with bias
			4) reg, the parameter for regularization

		This function calculates and returns negative log likelihood of the
		logistic regression with l2 regularization

		HINT:
			1) Recall the negative log likelihood function for logistic regression.
			2) Use a.sum() to find the summation of all entries in a numpy
			   array a
			3) Use np.linalg.norm to find the norm of a given vector
			4) Use np.log to caculate the log of each entry of the input array

		NOTE: please use the variable given for the final returned result, nll.
	"""

	"*** YOUR CODE HERE ***"
	inside = y * np.log(sigmoid(X @ W)) + (1 - y) * np.log(1 - sigmoid(X @ W))
	nll = -inside.sum() + reg/2 * np.linalg.norm(W) ** 2

	"*** END YOUR CODE HERE ***"
	return nll



def grad_descent(X, y, reg=0.0, lr=1e-4, eps=1e-6, max_iter=500, print_freq=20):
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
	# get the shape of the data, and initiate nll list
	m, n = X.shape
	nll_list = []

	# initialize the weight and its gradient
	W = np.zeros((n, 1))
	W_grad = np.ones_like(W)


	print('\n==> Running gradient descent...')

	# Start iteration for gradient descent
	iter_num = 0
	t_start = time.time()



	# HINT: Run the gradient descent algorithm followed steps below
	#	1) Calculate the negative log likelihood at each iteration and
	#	   append the value to nll_list
	#	2) Calculate the gradient for W
	#	3) Upgrade W
	#	4) Keep iterating while the number of iterations is less than the
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
		W_grad = grad_logreg(X, y, W, reg)
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
			t = t_end - t_start))

	return W, nll_list



#########################
#	    Step 1a`		#
#########################

def newton_step(X, y, W, reg=0.0):
	"""	This function takes in four arguments:
			1) X, the data matrix with dimension m x (n + 1)
			2) y, the label of the data with dimension m x 1
			3) W, a weight matrix with bias
			4) reg, the parameter for regularization

		This function calculates and returns the change of W according to
		the Newton's method

		HINT: get the result following the steps below
			1) Calculate the gradient of log likelihood, grad, with respect to W
			2) Use np.diag to create a diagonal matrix, with mu*(1-mu) being the
			   entries on the diagonal
			3) Calculate the Hessian matrix, H, of logistic regression following
			   the equation (you will need to use the diagonal matrix created)
			4) Using np.linalg.solve to solve for d in the equation Hd = -grad

		NOTE: Please use the variable given for final returned result, d.
	"""

	"*** YOUR CODE HERE ***"
	mu = sigmoid(X @ W)
	diag = np.diag(np.squeeze(np.asarray(np.multiply(mu, 1. - mu)))) ##LOOKED AT SOLUTION
	grad = grad_logreg(X, y, W, reg)
	H = X.T @ diag @ X + reg * np.eye(X.shape[1])
	d = np.linalg.solve(H, grad)
	"*** END YOUR CODE HERE ***"
	return d



def newton_method(X, y, reg=0.0, eps=1e-6, max_iter=20, print_freq=5):
	"""	This function takes in six arguments:
			1) X, the data with dimension m x (n + 1)
			2) y, the label of data with dimension m x 1
			3) reg, the parameter for regularization
			4) eps, the threshold of the norm for the gradients
			5) max_iter, the maximum number of iterations
			6) print_freq, the frequency of printing the report

		This function returns W, the optimal weight by Newton's Method,
		and nll_list, the corresponding learning objectives.
	"""
	# get the shape of the data, and initiate nll list
	m, n = X.shape
	nll_list = []

	# initialize the weight and its gradient
	W = np.zeros((n, 1))
	step = np.ones_like(W)

	print('==> Running Newton\'s method...')

	# Start iteration for gradient descent
	iter_num = 0
	t_start = time.time()


	# HINT: Run the gradient descent algorithm followed steps below
	#	1) Calculate the negative log likelihood at each iteration and
	#	   append the value to nll_list
	#	2) Calculate the gradient for W using newton_step defined above
	#	3) Upgrade W
	#	4) Keep iterating while the number of iterations is less than the
	#	   maximum and the gradient is larger than the threshold

	while iter_num < max_iter and np.linalg.norm(step) > eps:

		"*** YOUR CODE HERE ***"
		nll = NLL(X, y, W, reg)
		if np.isnan(nll):
			break
		nll_list.append(nll)
		W_change = newton_step(X, y, W, reg)
		W -= W_change

		"*** END YOUR CODE HERE ***"

		# Print statements for debugging
		if (iter_num + 1) % print_freq == 0:
			print('-- Iteration {} - negative log likelihood {: 4.4f}'.format(\
					iter_num + 1, nll))

		# Goes to the next iteration
		iter_num += 1



	# benchmark
	t_end = time.time()
	print('-- Time elapsed for running Newton\'s method: {t:2.2f} seconds'.format(\
			t = t_end - t_start))

	return W, nll_list




#########################
#		 Step 3			#
#########################

def predict(X, W):
	"""	This function takes in two arguments:
			1) W, a weight matrix with bias
			2) X, the data with dimension m x (n + 1)

		This function calculates and returns the predicted label.

		NOTE: You don't need to change this function.
	"""
	mu = sigmoid(X @ W)
	return (mu >= 0.5).astype(int)



def get_description(X, y, W):
	"""	This function takes in three arguments:
			1) X, the data matrix with dimension m x (n + 1)
			2) y, the label of the data with dimension m x 1.
			3) W, the weight matrix with bias and dimension (n + 1) x 1

		This function calculates and returns the accuracy, precision,
		recall and F-1 score of the prediction.

		HINT:
			1) Get the predict lables using predict defined above
			2) Accuracy = probability of correct prediction
			3) Precision = probability of true label being 1 given that the
			   predicted label is 1
			4) Recall = probablity of predicted label being 1 given that the
			   true label is 1
			5) F-1 = 2*p*r / (p + r), where p = precision and r = recall

		NOTE: Please use the variable given for final returned results.
	"""

	"*** YOUR CODE HERE ***"
	y_hat = predict(X, W)
	accurate_count = 0
	prec_num_pred_label, prec_num_true_label = 0, 0
	rec_num_pred_label, rec_num_true_label = 0, 0
	for i in range(y.shape[0]):
		if y_hat[i] == y[i]: accurate_count += 1
		if y_hat[i] == 1:
			prec_num_pred_label += 1
			if y[i] == 1: prec_num_true_label += 1
		if y[i] == 1:
			rec_num_true_label += 1
			if y_hat[i] == 1: rec_num_pred_label += 1
	accuracy = accurate_count / y.shape[0]
	precision = prec_num_true_label / prec_num_pred_label
	recall = rec_num_pred_label / rec_num_true_label
	f1 = (2 * precision * recall) / (precision + recall)
	"*** END YOUR CODE HERE ***"
	return accuracy, precision, recall, f1




def plot_description(X_train, y_train, X_test, y_test):
	"""	This function takes in four arguments:
			1) X_train, the training data with dimension m x (n + 1)
			2) y_train, the label of training data with dimension m x 1
			3) X_val, the validation data with dimension m x (n + 1)
			4) y_val, the label of validation data with dimension m x 1

		This function plots the accuracy/precision/recall/F-1 score versus
		lambda and returns the lambda that maximizes accuracy.
	"""
	reg_list = []
	a_list = []
	p_list = []
	r_list = []
	f1_list = []


	# HINT:
	# 	1) First, generate/create a list of different lambda
	#   2) For each lambda, run gradient descent and obtain the optimal weights
	# 	3) Get accuracy, precision, recall and f1 with the data and W_opt
	#	   obtained, and append those into the corresponding list

	"*** YOUR CODE HERE ***"
	reg_list = [0, .01, .1, .25, .5, 1, 2, 5, 10, 20, 50, 100]
	for l in reg_list:
		W_opt, nll = grad_descent(X_train, y_train, l)
		accuracy, precision, recall, f1 = get_description(X_test, y_test, W_opt)
		a_list.append(accuracy)
		p_list.append(precision)
		r_list.append(recall)
		f1_list.append(f1)
	"*** END YOUR CODE HERE ***"


	# Generate plots
	# plot accurary versus lambda
	a_vs_lambda_plot, = plt.plot(reg_list, a_list)
	plt.setp(a_vs_lambda_plot, color = 'red')

	# plot precision versus lambda
	p_vs_lambda_plot, = plt.plot(reg_list, p_list)
	plt.setp(p_vs_lambda_plot, color = 'green')

	# plot recall versus lambda
	r_vs_lambda_plot, = plt.plot(reg_list, r_list)
	plt.setp(r_vs_lambda_plot, color = 'blue')

	# plot f1 score versus lambda
	f1_vs_lambda_plot, = plt.plot(reg_list, f1_list)
	plt.setp(f1_vs_lambda_plot, color = 'yellow')

	# Set up the legend, titles, etc. for the plots
	plt.legend((a_vs_lambda_plot, p_vs_lambda_plot, r_vs_lambda_plot, \
		f1_vs_lambda_plot), ('accuracy', 'precision', 'recall', 'F-1'),\
		 loc = 'best')
	plt.title('Testing descriptions')
	plt.xlabel('regularization parameter')
	plt.ylabel('Metric')
	plt.savefig('hw4pr2a_description.png', format = 'png')
	plt.close()

	print('==> Plotting completed.')


	"*** YOUR CODE HERE ***"
	max_index = reg_list.index(max(reg_list))
	reg_opt = reg_list[max_index]
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

	# data frame
	df_train = data.df_train
	df_test = data.df_test

	# training data
	X_train = data.X_train
	y_train = data.y_train

	# test data
	X_test = data.X_test
	y_test = data.y_test



	# =============STEP 1: Logistic regression=================
	print('\n\n==> Step 1: Running logistic regression...')

	# splitting data for logistic regression
	# NOTE: for logistic regression, we only want images with label 0 or 1.
	df_train_logreg = df_train[df_train.label <= 1]
	df_test_logreg = df_test[df_test.label <= 1]

	# training data for logistic regression
	X_train_logreg = np.array(df_train_logreg[:][[col for \
		col in df_train_logreg.columns if col != 'label']]) / 256.
	y_train_logreg = np.array(df_train_logreg[:][['label']])

	# testing data for logistic regression
	X_test_logreg = np.array(df_test_logreg[:][[col for \
		col in df_test_logreg.columns if col != 'label']]) / 256.
	y_test_logreg = np.array(df_test_logreg[:][['label']])

	# stacking a column of 1's to both training and testing data
	X_train_logreg = np.hstack((np.ones_like(y_train_logreg), X_train_logreg))
	X_test_logreg = np.hstack((np.ones_like(y_test_logreg), X_test_logreg))


	# ========STEP 1a: Gradient descent=========
	# NOTE: Fill in the code in grad_logreg, NLL and grad_descent for this step

	print('\n==> Step 1a: Running gradient descent...')
	W_gd, nll_list_gd = grad_descent(X_train_logreg, y_train_logreg, reg = 1e-6)


	# ========STEP 1b: Newton's method==========
	# NOTE: Fill in the code in newton_step and newton_method for this step

	print('\n==> Step 1b: Running Newton\'s method...')
	W_newton, nll_list_newton = newton_method(X_train_logreg, y_train_logreg, \
		reg = 1e-6)



	# =============STEP 2: Generate convergence plot=================
	# NOTE: You DO NOT need to fill in any additional helper function for this
	# 		step to run. This step uses what you implemented for the previous
	#		two steps to plot.
	print('\n==> Step 2: Generate Convergence Plot...')
	print('==> Plotting convergence plot...')

	# set up the style for the plot
	plt.style.use('ggplot')

	# plot gradient descent and newton's method convergence plot
	nll_gd_plot, = plt.plot(range(len(nll_list_gd)), nll_list_gd)
	plt.setp(nll_gd_plot, color = 'red')

	nll_newton_plot, = plt.plot(range(len(nll_list_newton)), nll_list_newton)
	plt.setp(nll_newton_plot, color = 'green')

	# add legend, titles, etc. for the plots
	plt.legend((nll_gd_plot, nll_newton_plot), \
		('Gradient descent', 'Newton\'s method'), loc = 'best')
	plt.title('Convergence Plot on Binary MNIST Classification')
	plt.xlabel('Iteration')
	plt.ylabel('NLL')
	plt.savefig('hw4pr2a_convergence.png', format = 'png')
	plt.close()

	print('==> Plotting completed.')



	# =============STEP 3: Generate accuracy/precision plot=================
	# NOTE: Fill in the code in get_description and plot_description for this Step

	print('\nStep 3: ==> Generating plots for accuracy, precision, recall, and F-1 score...')

	# Plot the graph and obtain the optimal regularization parameter
	reg_opt = plot_description(X_train_logreg, y_train_logreg, \
		X_test_logreg, y_test_logreg)

	print('\n==> Optimal regularization parameter is {:4.4f}'.format(reg_opt))
