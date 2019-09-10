
import numpy as np


class Network:

	# Input parameter
	#   sizes: list in which each element is the number of neurons in each layer.
	def __init__(self, sizes):
		self.num_layers = len(sizes)
		self.sizes = sizes
		self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
			# There is one bias for each node, except for those in the input layer.
		self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
			# Except for the input layer...
			# weights is a list of layers. Each layer has a list of nodes.
			# Each node has a list of input weights. There is a weight for each of the nodes from the previous layer.
			# E.g., weights[1] is a list containing the weights connecting the second layer to the third.

		#print(sizes[1:])
		# for elem in self.biases:
		# 	print(elem)
		# for elem in self.weights:
		# 	print(elem)

		# size_test = [4, 5, 6, 7, 3]
		# for elem in zip(size_test[:-1], size_test[1:]):
		# 	print(elem)

### Miscellaneous functions
def sigmoid(a):
	z = 3
	print(np.exp(z))

	z = np.asarray([1, 3, 4])
	print(-z)

	y = np.exp(z)
	print(y)
	print(1 + y)
	print(1/(1 + y))


	# Operations on a numpy array are applied individually to each element.
	# So this will return a list of elements with the operations below
	# 	applied individually to each element.
	# The list is for a whole layer, consists of the activation values for
	# 	each node.
	return 1.0/(1.0 + np.exp(-a))
