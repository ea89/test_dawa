"""Sample code for using the Data- and Workload-Aware (DAWA) algorithm and its sub-algorithms."""

import numpy

import dawa
import l1partition

def main():
	print ("""In this sample, we use the following data vector x
	x = [2, 3, 8, 1, 0, 2, 0, 4, 2, 4]
	""")
	x = [2, 3, 8, 1, 0, 2, 0, 4, 2, 4]
	
	print ("""Compute a data-aware histogram of x using the L1 parition algorithm, with the following parameters:
	epsilon: 1
	amount of privacy budget used to compute the partition: 25%""")
	# Use the L1 parition algorithm to divide x into buckets
	# Fix the random seeds to get the consistent results in two runs
	numpy.random.seed(10)
	
	print("The buckets in the histogram are:")
	# Run the L1 partition algorithm but only ask for the buckets
	print (l1partition.L1partition(x, 1, 0.25, gethist=True))

	numpy.random.seed(10)
	print("The estimated data vector is:")
	# Run the L1 partition algorithm again, and ask for the estimated data vector
	print ("\t", l1partition.L1partition(x, 1, 0.25))

	
if __name__ == "__main__":
	main()