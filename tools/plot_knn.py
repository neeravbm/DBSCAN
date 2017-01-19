import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
	df = pd.read_csv(sys.argv[1], sep=',', header=None, names=['id', 'dist'])
	print df.shape
	df.plot(x='id', y='dist')
	plt.show()

if __name__ == '__main__':
	main()