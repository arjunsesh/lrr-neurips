import numpy as np
import pdb 
from tqdm import tqdm
import torch as t
import torch.nn.functional as F
import torch.nn as nn
from scipy.stats import chi2, truncnorm
import sys
import cdm_pytorch as cp
import matplotlib.pyplot as plt

def gen_order(n):
	order = np.arange(n)
	np.random.shuffle(order)

	return order

def process_order(n, B=10, order_type='evenly', k=None):
	if order_type == 'evenly':
		gamma = np.exp(2*B/(n-1))
		p = gamma**(np.arange(n))
		p /= p.sum()
	elif order_type == 'uniform':
		p = np.ones(n)
		p /= p.sum()
	elif order_type == 'extreme':
		gamma = np.zeros(n)
		if k is not None:
			gamma[:int(k)] = np.exp(-B)
			gamma[int(k):] = np.exp(B)
		else:
			gamma[:n-1] = np.exp(-B)
			gamma[n-1:] = np.exp(B)
		p = gamma/gamma.sum()

	return p

def gen_order_weighted(n, B=10, order_type='evenly', k=None):
	if order_type == 'evenly' or order_type == 'uniform' or order_type == 'extreme':
		p = process_order(n, B, order_type, k)
	elif order_type == 'constant':
		return np.arange(n)
	elif order_type == 'last_dom':
		return np.append(n-1, np.random.choice(n-1, size=n-1, replace=False))
	elif order_type == 'k_dom':
		k = n-1 if k is None else k
		rank = list(range(n))
		rank.remove(k)
		return np.append(k, np.random.choice(rank, size=n-1, replace=False))

	return np.random.choice(n, size=n, replace=False, p = p)

def gen_matrix(order):
	order = order[::-1]
	vec = np.zeros([len(order),1])
	matrix = 0
	vec[order[0]] = 1 
	for i in range(1,len(order)):
		vec[order[i]] = 1
		matrix += np.diag(vec[:,0]) - (1/(i+1))*vec.dot(vec.T)

	return len(order)*matrix

def gen_choices_from_rank(order):
	order = order[::-1]
	vec = np.zeros([len(order),1])
	choice_sets = np.zeros([len(order)-1, len(order)])
	choices = np.zeros([len(order)-1, len(order)])
	vec[order[0]] = 1 
	for i in range(1,len(order)):
		vec[order[i]] = 1
		choice_sets[i-1] = vec[:,0]
		choices[i-1, order[i]] = 1


	return choice_sets, choices


def gen_matrix2(order):
	order = order[::-1]
	vec = np.zeros([len(order),1])
	matrix = 0
	vec[order[0]] = 1 
	for i in range(1,len(order)):
		vec[order[i]] = 1
		matrix += (i+1)*np.diag(vec[:,0]) - vec.dot(vec.T)

	return matrix


def gen_L(m,n, B=None, order_type='evenly', k=None):
	L = 0
	for i in tqdm(range(m)):
		order = gen_order_weighted(n,B, order_type,k=k) if B is not None else gen_order(n) 
		L += gen_matrix(order)

	return L/(m*(n-1))

def gen_min_wij(m,n, B=None, order_type='evenly', k=None):
	L = 0
	for i in tqdm(range(m)):
		order = gen_order_weighted(n,B, order_type,k=k) if B is not None else gen_order(n) 
		L += gen_matrix2(order)
	np.fill_diagonal(L,-np.inf)
	
	return np.min(-L[np.triu_indices(n)])/(m*(n-1))

def gen_O_J(m,n, B=None, order_type='evenly', k=None):
	O, J = [], []
	for i in tqdm(range(m)):
		order = gen_order_weighted(n,B, order_type,k=k) if B is not None else gen_order(n) 
		o, j = gen_choices_from_rank(order)
		O.append(o)
		J.append(j)

	O = np.concatenate(O,0)
	J = np.concatenate(J,0)

	return O, J

def gen_O_J_v2(m,n,p):
	# Given a fixed p, it will generate lots of choices.
	O, J = [], []
	
	for i in tqdm(range(m)):
		order = np.random.choice(n, size=n, replace=False, p = p)
		o, j = gen_choices_from_rank(order)
		O.append(o)
		J.append(j)

	O = np.concatenate(O,0)
	J = np.concatenate(J,0)

	return O, J

def resample_choice(o, p):
	j = np.zeros(o.shape)
	o_nnz = o.nonzero()[0]
	j[o_nnz[np.random.choice(len(o_nnz), 1, p=p[o_nnz]/p[o_nnz].sum())]] = 1
	return j

def resample_J(O, B=None, order_type='evenly', k=None):
	J = []
	_, n = O.shape
	for o in O:
		p = process_order(n, B, order_type=order_type)
		J.append(resample_choice(o, p)[None,:])

	J = np.concatenate(J,0)
	return J

def simple_theta_estimator(m, gamma=None):
	if gamma is None:
		gamma = np.array([1/3, 1/3, 1/3])
	theta = np.log(gamma)
	theta -= np.mean(theta)
	alpha = np.zeros(np.shape(gamma))
	alpha[0] = .5*gamma[0]/np.sum(gamma[[0,1]])
	alpha[2] = .5*gamma[2]/np.sum(gamma[[1,2]])
	alpha[1] = 1 - np.sum(alpha[[0,2]])
	alpha_hat = np.bincount(np.random.choice(3, m, p=alpha))/m

	theta_hat = np.zeros(np.shape(gamma))
	theta_hat[1] = -np.log(1 + 2*alpha_hat[0]/(1 - 2*alpha_hat[0]) + 2*alpha_hat[2]/(1 - 2*alpha_hat[2]))
	theta_hat[0] = np.log(2*alpha_hat[0]/(1 - 2*alpha_hat[0])) + theta_hat[1]
	theta_hat[2] = np.log(2*alpha_hat[2]/(1 - 2*alpha_hat[2])) + theta_hat[1]

	theta_hat -= np.mean(theta_hat)
	return alpha, alpha_hat, theta, theta_hat

def simple_expectation(T=1000, m=1000, gamma=None):
	alpha_hat = []; theta_hat = []
	for t in tqdm(range(T)):
		a,a_h,th,th_h  = theta_estimator(m, gamma)
		alpha_hat.append(a_h)
		theta_hat.append(th_h)

	return a - np.mean(alpha_hat,0), th - np.mean(theta_hat,0)

def softmax_hessian(x):
	# make x a column vector n x 1
	a = np.exp(x)
	a /= a.sum()
	return (np.diag(a[:,0]) - a.dot(a.T))

### Methods for L2 Error Plot
def gen_datasets(n=6, num_rankings=1000, num_datasets=20, B=1.5, random_state=8080):
	theta = truncnorm.rvs(-B, B, loc=0, scale=1, size=n, random_state=random_state)
	theta = np.array(theta); theta -= theta.mean()
	print(f'theta: {theta}')
	p = np.exp(theta)
	p /= p.sum()

	data = [gen_O_J_v2(num_rankings,n,p) for idx in range(num_datasets)]

	return theta, data

def fit_functions(data, num_fits, start_point, end_point, logspace=True):
	param_store = []
	gv_store = []

	if logspace:
		increment_sizes=np.int32(np.logspace(start_point, end_point, num_fits))
	else:
		increment_sizes=np.int32(np.linspace(start_point, end_point, num_fits))
	
	for dataset in data:
	    O, J = dataset
	    _,n = O.shape
	    param_store.append([])
	    gv_store.append([])
	    for fit in range(num_fits):
	        O_temp = O[:(n-1)*increment_sizes[fit]]
	        J_temp = J[:(n-1)*increment_sizes[fit]]
	        
	        model, _, gv = cp.l2err_run(O_temp,J_temp, Model=cp.MNL, epochs=500, lr=1e-2)
	        opt = model.logits.weight.detach().numpy()[:-1] # remove padding

	        param_store[-1].append(opt[:,0]-opt.mean())
	        gv_store[-1].append(gv)

	param_store = np.array(param_store)

	return param_store, gv_store, increment_sizes

def generate_PL_error_plot(n=6, L=10000, num_datasets=20, B = 1.5, num_fits=20):
	start_point = np.log10(L/(100))
	end_point = np.log10(L)

	theta, data = gen_datasets(n,L, num_datasets, B)
	param_store, gv_store, gl = fit_functions(data, num_fits, start_point, end_point, logspace=True)
	L2_err = np.sum((param_store - theta[None, None, :])**2, -1).T # num_datasets x num_fits

	plt.clf()
	plt.loglog(gl,L2_err)
	plt.loglog(gl,70/np.array(gl), 'k',linewidth=2, linestyle='--')
	plt.xlabel('Rankings')
	plt.ylabel('Squared $\ell_2$ Error')
	# plt.title('PL Convergence, n=6, 20 lines, 20 increments')
	#plt.legend(range(num_datasets))
	#plt.legend(['LRT CDM', 'LRT Choice System',])#'LRT CDM Adj','LRT Choice System Adj'])
	plt.savefig('l2errPLloglog.pdf',bbox_inches='tight')