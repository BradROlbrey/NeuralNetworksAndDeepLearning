import random

import numpy as np


class Network:

	# Input parameter
	#   sizes: list in which each element is the number of neurons in each layer.
	def __init__(self, sizes):
		self.num_layers = len(sizes)
		self.sizes = sizes
		self.biases = [np.random.randn(y) for y in sizes[1:]]
			# There is one bias for each node, except for those in the input layer.
		self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
			# Except for the input layer...
			# weights is a list of layers. Each layer has a list of nodes.
			# Each node has a list of input weights. There is a weight for each of the nodes from the previous layer.
			# E.g., weights[1] is a list containing the weights connecting the second layer to the third.


	# Returns the output of the neural net given an input, a.
	def feedforward(self, a):
		# The zip pairs the weights and biases for each layer
		#	so we can get the output values for this layer.
		# We loop to feed forward all the way to the output.
		for w, b in zip(self.weights, self.biases):
			a = sigmoid(np.dot(w, a) + b)
		return a

	"""Train the neural network using mini-batch stochastic
	gradient descent."""
	# Input params:
	#	training_data: a list of tuples "(x, y)" representing the
	#		training inputs and the desired outputs.
	#	epochs: number of "epochs" to train for. One epoch involves training over
	#		the entire training data set.
	#	batch_size: size of the batches to divide training_data into.
	#	eta: learning rate.
	def SGD(self, training_data, epochs, batch_size, eta):

		n = len(training_data)
		for epoch in range(epochs):

			# Get the mini batches
			random.shuffle(training_data)
			mini_batches = [
				training_data[k : k + batch_size]
				for k in range(0, n, batch_size)  # increment by batch_size
			]

			# Train on each mini batch
			for mini_batch in mini_batches:
				self.train_on_mini_batch(mini_batch, eta)

			print("Epoch {} complete".format(epoch))


	"""Update the network's weights and biases by applying
	gradient descent using backpropagation to a single mini batch."""
	def train_on_mini_batch(self, mini_batch, eta):
		return



### Miscellaneous functions

# Used by Network.feedforward
# 	Takes as input wa+b for each node and applies the sigmoid
# 		function to each element.
# Returns a list of the final activation/output values for each node in
# 	the layer.
def sigmoid(a):
	# Operations on a numpy array are applied individually to each element.
	# So this will return a list of elements with the operations below
	# 	applied individually to each element.
	# The list is for a whole layer, consists of the output values for
	# 	each node.
	return 1.0/(1.0 + np.exp(-a))
