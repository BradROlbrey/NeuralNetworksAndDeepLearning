
from Network import Network, sigmoid


def main():
	print("Hello, World!")
	net = Network([2, 3, 2])
	net.feedforward([1,2,3])


if __name__ == '__main__':
	main()
