import numpy as np
import os,sys
from functools import reduce

def RS_choices(L,n):
	"""
	Encodes RS(sigma) choices for a list of rankings with a
	stack of indicator vectors for the choice sets with a 'one-hot'
	encoding of choices

	Args:
	L- list of (possibly partial) rankings
	n- number of alternatives in the

	Returns:
	X- X[j,:] is an indicator vector for the j-th choice set
	Y- Y[j,:] is a 'one-hot' encoding of the j-th choice
	"""
	#computes the total number of choices
	m = reduce(lambda x,y: x+y,map(len,L))
	X = np.zeros((m,n)); Y = np.empty(m)
	i = 0; j = 0
	for sigma in L:
		S = list(range(n))
		for i in range(len(sigma)):
			X[j,S] = 1
			assert np.sum(X[j,:])>0
			Y[j]=sigma[i]
			S.remove(sigma[i])
			j+=1
	return X,Y.astype(int)

def RE_choices(L,n):
	"""
	computes the RE choices for a list of rankings L by computing the
	RS choices on a reversal of the rankings
	"""
	return RS_choices(map(lambda s: s[::-1],L),n,alpha)

def scrape_soi(filename):
	"""
	scrapes an soi file into a list of (possibly partial) rankings

	Args:
	filename- name of soi file to scrape
	"""
	L = []
	with open(filename,'r') as f:
		N = int(next(f))
		if N<2:
			return [],0
		#sometimes the first candidate is labeled '1', sometimes '0'
		offset = int(next(f)[0])
		for _ in range(N):
			next(f)
		for line in f:
			l = line[:-1].split(',')
			count = int(l[0])
			sig = map(lambda x: int(x)-offset, l[1:])

			for _ in range(count):
				#some election data had repeated "write-in" markers
				L.append(list(sig))

	return L,N

def scrape_soc(filename):
	"""
	scrapes an soi file into a list of complete rankings

	Args:
	filename- name of soc file to scrape
	"""
	L = []
	with open(filename,'r') as f:
		N = int(next(f))
		#if N > 100:
		#	#these are too big
		#	return [],0
		#sometimes the first candidate is labeled '1', sometimes '0'
		offset = int(next(f)[0])
		for _ in range(N):
			next(f)
		for line in f:
			l = line.split(',')
			count = int(l[0])
			sig =[]
			i = 1
			sig = map(lambda k: int(k)-offset,l[1:])
			for _ in range(count):

				#some election data had repeated "write-in" markers
				L.append(list(sig))

	return L,N
