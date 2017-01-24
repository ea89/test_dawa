***********************************************************************************************
Data- and Workload-Aware Algorithm for Range Queries Implementation
Version 1.0, Last updated: 11/28/2013
Please contact chaoli [at] cs.umass.edu and miklau [at] cs.umass.edu for comments or questions.
***********************************************************************************************

Overview:
---------
The included source code implements the Data- and Workload-Aware (DAWA) algorithm for range query answering under differential privacy [1].

Install:
----------------------
Run setup.sh from the base directory

Usage:
-----------------------
The usage examples can be found in example.py

To run the complete DAWA algorithm:

	import dawa
	estx = dawa.dawa(Q, x, epsilon, ratio)
	
	Q - a list of queries: [q1, q2, ...]. 
		Each query is a list of triplets: [ [w1, lb1, rb1], ... [wk, lbk, rbk] ],
		which represents the query w1 ( x[lb1] + ... + x[rb1] ) + ... + wk ( x[lbk] + ... + x[rbk] )
		
	epsilon - double. Total private budget

	x - a list of numerical values: [x1, ..., xn], which is the vector that represents the underlying database.

	ratio - double in [0, 1). use ratio*epsilon for partition computation and (1-ratio)*epsilon for querying
			the count in each partition. If ratio==0, the algorithm will not call partition computation. Instead,
			it assumes each cell is in its own bucket.
			
	estx - an estimated data vector
	
To run the noisy partition algorithm:

	import l1partition
	estx = l1partition.L1partition(x, epsilon, ratio)
		
To run the noisy partition algorithm with bucket size constrained to powers of 2 (the first step of DAWA):

	import l1partition
	estx = l1partition.L1partition_approx(x, epsilon, ratio)

To run the data independent workload adapting algorithm (the second step of DAWA):
	import dawa
	estx = dawa.dawa(Q, x, epsilon, 0)


Requirements: 
------------- 

Tested with: Python 2.6.6/2.7.1, Numpy 1.9.1, Scipy 0.11.0, Swig 2.0.4


Related publications:
---------------------

[1] Chao Li, Michael Hay, Gerome Miklau, and Yue Wang, A Data- and Workload-Aware Algorithm for Range Queries Under Differential Privacy. To be appear in PVLDB 2014