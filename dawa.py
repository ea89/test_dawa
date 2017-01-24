import sys
import itertools

import numpy as np
from scipy.optimize import nnls
from scipy.linalg import block_diag

sys.path.append("./cutils/")

from l1partition import L1partition_approx


def _greedyHierByLv(fullQ, n, offset, depth = 0, branch = 2, withRoot = False):
	if n == 1:
		return np.linalg.norm(fullQ[:, offset], 2)**2, np.array([[1.0]], dtype=np.float32), np.array([1.0]), [[offset, offset]]
		
	Q = fullQ[:, offset:offset+n]
	if (np.min(Q, axis=1) == np.max(Q, axis=1)).all():
		mat = np.zeros([n, n], dtype=np.float32)
		mat.fill(1.0 / n**2)
		return np.linalg.norm(Q[:,0], 2)**2, mat, np.array([1.0]), [[offset, offset+n-1]]
		
	if n <= branch:
		bound = zip(range(n), range(1,n+1))
	else:
		rem = n % branch
		step = (n-rem) / branch
		swi = (branch-rem) * step
		sep = range(0, swi, step) + range(swi, n, step+1) + [n]
		bound = zip(sep[:-1], sep[1:])
		
	serr, sinv, sdist, sq = zip(*map(lambda c: _greedyHierByLv(fullQ, c[1]-c[0], offset+c[0], depth = depth+1, branch = branch), bound))
	invAuList = map(lambda c: c.sum(axis=0), sinv)
	invAu = np.hstack(invAuList)
	k = invAu.sum()
	m1 = sum(map(lambda rng, v: np.linalg.norm(np.dot(Q[:, rng[0]:rng[1]], v), 2)**2, bound, invAuList))
	m = np.linalg.norm(np.dot(Q, invAu), 2)**2
	sumerr = sum(serr)
	
	if withRoot:
		return sumerr, block_diag(*sinv), np.hstack([[0], np.hstack(sdist)]), [[offset, offset+n-1]] + list(itertools.chain(*sq))
		
	granu = 100
	decay = 1.0 / ( branch**(depth / 2.0))
	err1 = np.array(range(granu, 0, -1), dtype=float)**2
	err2 = np.array(range(granu), dtype=float)**2 * decay
	toterr = 1.0/err1 * (sumerr - ((m-m1)*decay+m1) * err2 / (err1+err2*k))
	
	err = toterr.min() * granu**2
	perc = 1 - np.argmin(toterr) / float(granu)
	inv = (1.0/perc)**2 * (block_diag(*sinv) - (1-perc)**2 / ( perc**2 + k * (1-perc)**2 ) * np.dot(invAu.reshape([n, 1]), invAu.reshape([1, n])))
	dist = np.hstack([[1-perc], perc*np.hstack(sdist)])
	return err, inv, dist, [[offset, offset+n-1]] + list(itertools.chain(*sq))



def dawa(Q, x, epsilon, ratio=0.25, branch = 2):
	"""Data- and Workload-aware algorithm.
		Given a query set Q and data vector x, output an estimated data vector that answers queries in Q with low noise
		
	Common Args:
		Q - a list of queries: [q1, q2, ...]. 
			Each query is a list of triplets: [ [w1, lb1, rb1], ... [wk, lbk, rbk] ],
			which represents the query w1 ( x[lb1] + ... + x[rb1] ) + ... + wk ( x[lbk] + ... + x[rbk] )
		epsilon - double. Total private budget
		x - a list of numerical values
		ratio - double in [0, 1). use ratio*epsilon for partition computation and (1-ratio)*epsilon for querying
				the count in each partition. If ratio==0, the algorithm will not call partition computation. Instead,
				it assumes each cell is in its own bucket.
		
	Experimental Args (Do not change unless necessary):
		branch - int >= 2. branching factor of the H-tree
	"""
	n = len(x)
	if ratio > 0:
		hist = L1partition_approx(x, epsilon, ratio, gethist=True)
	else:
	    hist = [[c, c] for c in range(n)]
		
	n2 = len(hist)
	cnum = range(0, len(Q), n)
	QtQ = np.zeros([n2, n2])
	for c0 in cnum:
		nrow = min(len(Q)-c0, n)
		Q0mat = np.zeros([nrow, n])
		for c in range(nrow):
			for wt, lb, rb in Q[c+c0]:
				Q0mat[c, lb:rb+1] = wt
				
		Qmat = np.zeros([nrow, n2])
		for c in range(n2):
			lb, rb = hist[c]
			Qmat[:, c] = Q0mat[:, lb:rb+1].mean(axis=1)
			
		QtQ += np.dot(Qmat.T, Qmat)
		
	err, inv, dist, query = _greedyHierByLv(QtQ, n2, 0, branch = branch, withRoot=False)
	inv = inv / ((1.0-ratio)**2)
	qmat = []
	y2 = []
	for c in range(len(dist)):
		if dist[c] > 0:
			lb, rb = query[c]
			currow = np.zeros(n2)
			currow[lb:rb+1] = dist[c]*(1-ratio)
			qmat.append(currow)
			y2.append(sum(x[hist[lb][0]:hist[rb][1]+1]) * dist[c]*(1-ratio))
		
	qmat = np.array(qmat)
	y2 += np.random.laplace(0.0, 1.0/epsilon, len(y2))
	
	estv = np.dot(inv, np.dot(qmat.T, y2)) 
		
	estx = np.zeros(n)
	for c in range(n2):
		estx[hist[c][0]:hist[c][1]+1] = estv[c] / float(hist[c][1] - hist[c][0] + 1)
		
	return estx

