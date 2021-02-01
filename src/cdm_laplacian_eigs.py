import numpy as np
import pdb 
from tqdm import tqdm
import torch as t
import torch.nn.functional as F
import torch.nn as nn
from scipy.stats import chi2, truncnorm
import sys
import pickle as pkl
import cdm_pytorch as cp 
import matplotlib.pyplot as plt

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

def gen_CDM_ranking(U):
	order = []
	vec = np.ones(U.shape[0], dtype=np.bool)
	for idx in range(U.shape[0]-1):
		p = np.exp(U[:,vec][vec,:].sum(-1))
		p /= p.sum()
		order.append(vec.nonzero()[0][np.random.choice(len(p),p=p)])
		vec[order[-1]] = 0
	order.append(vec.nonzero()[0][0])

	return np.array(order)

def gen_O_J_v2(m,n,U):
	# EDITED FOR CDM
	# Given a fixed CDM U, it will generate lots of choices.
	O, J = [], []
	
	for i in tqdm(range(m)):
		order = gen_CDM_ranking(U)
		o, j = gen_choices_from_rank(order)
		O.append(o)
		J.append(j)

	O = np.concatenate(O,0)
	J = np.concatenate(J,0)

	return O, J

def set_to_edge(o):
	n = len(o)
	o_nnz = o.nonzero()[0]

	L_list = np.zeros([n*(n-1), len(o_nnz)])
	for idx_nz, idx in enumerate(o_nnz):
		L_list[idx*(n-1):(idx+1)*(n-1),idx_nz] = np.delete(o, idx)
	L_sum = L_list.sum(-1,keepdims=True)
	#pdb.set_trace()
	return L_list.dot(L_list.T) - (1/(len(o_nnz)))*L_sum.dot(L_sum.T)

### Methods for L2 Error Plot
def vectorize(U):
    n,_=U.shape
    u=np.array([U[i,j] for i in range(n) for j in range(n) if i!=j])
    return (u-u.mean())

def gen_datasets(n=6, num_rankings=1000, num_datasets=20, B=1.5, theta=None, d=None, random_state=8080):
	if theta is not None:
		theta = np.array(theta)
		theta -= theta.mean()
		U = -np.ones([n,1]).dot(theta); np.fill_diagonal(U,0); 
	elif d is not None:
		T = truncnorm.rvs(-B, B, loc=0, scale=1, size=(n,d), random_state=random_state)
		C = truncnorm.rvs(-B, B, loc=0, scale=1, size=(n,d), random_state=random_state)
		T /= np.sqrt(d*(n-1))
		C /= np.sqrt(d*(n-1))
		U = T.dot(C.T)

		np.fill_diagonal(U,0); U -= U.sum()/(n*(n-1)); np.fill_diagonal(U,0)

	else:
		U = truncnorm.rvs(-B, B, loc=0, scale=1, size=(n,n), random_state=random_state)
		np.fill_diagonal(U,0); U /= n-1; U -= U.sum()/(n*(n-1)); np.fill_diagonal(U,0)
	
	print(f'U: {U}')
	
	data = [gen_O_J_v2(num_rankings,n,U) for idx in range(num_datasets)]

	u = vectorize(U)

	return u, data

def fit_functions(data, num_fits, start_point, end_point, d=None, logspace=True):
	param_store = []
	gv_store = []

	if logspace:
		if num_fits == 1:
			increment_sizes = [int(10**end_point)]
		else:
			increment_sizes=np.int32(np.logspace(start_point, end_point, num_fits))
	else:
		if num_fits == 1:
			increment_sizes = [end_point]
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
	        #print(O_temp.shape)
	        # opt, gv = mu.CDM_optimize(O_temp,J_temp, num_steps=1000, step_size=1e-2)
	        if d is not None:
	        	model, _, gv = cp.l2err_run(O_temp,J_temp, Model=cp.CDM, epochs=1000, lr=1e-2, embedding_dim=d)
	        	opt_T = model.target_embedding.weight.detach().numpy()[:-1, :] # remove padding
	        	opt_C = model.context_embedding.weight.detach().numpy()[:-1, :] # remove padding
	        	opt = opt_T.dot(opt_C.T)
	        else:
	        	model, _, gv = cp.l2err_run(O_temp,J_temp, Model=cp.FullCDM, epochs=1000, lr=1e-2)
	        	opt = model.U.weight.T.detach().numpy()[:-1,:-1] # remove padding

	        
	        param_store[-1].append(opt)
	        gv_store[-1].append(gv)

	param_store = np.array([[vectorize(opt) for opt in param_set] for param_set in param_store])

	return param_store, gv_store, increment_sizes


def generate_CRS_err_plot(n=6, L=10000, num_datasets=20, B = 1.5, num_fits=20):
	start_point = np.log10(L/(100)); end_point = np.log10(L)
	u, data = gen_datasets(n,L, num_datasets, B)
	param_store, gv_store, gl = fit_functions(data, num_fits, start_point, end_point, logspace=True)
	L2_err = np.sum((param_store - u[None, None, :])**2, -1).T

	plt.clf()
	plt.loglog(gl,L2_err)
	plt.loglog(gl,3000/np.array(gl), 'k',linewidth=2, linestyle='--')
	plt.xlabel('Rankings')
	plt.ylabel('Squared $\ell_2$ Error')

	plt.savefig('l2errCRSloglog.pdf',bbox_inches='tight')

def generate_CRS_on_PL_err_plot(n=6, L=10000, num_datasets=20, B = 1.5, num_fits=20):
	start_point = np.log10(L/(100)); end_point = np.log10(L)
	# theta for PL, same as PL from random_state 8080
	theta = np.array([[ 0.50883968,  1.15971874, -0.5272781,  -0.13336062,  0.14942428,  0.37038735]])
	u, data = gen_datasets(n,L, num_datasets, B, theta=theta)
	param_store, gv_store, gl = fit_functions(data, num_fits, start_point, end_point, logspace=True)
	L2_err = np.sum((param_store - u[None, None, :])**2, -1).T

	plt.clf()
	plt.loglog(gl,L2_err)
	plt.loglog(gl,3000/np.array(gl), 'k',linewidth=2, linestyle='--')
	plt.xlabel('Rankings')
	plt.ylabel('Squared $\ell_2$ Error')

	plt.savefig('l2errCRSonPLloglog.pdf',bbox_inches='tight')

def generate_CRS_error_plot_variousn(n_list=[6, 9, 12, 16], L=10000, num_datasets=20, B = 1.5, num_fits=20):
	start_point = np.log10(L/(100)); end_point = np.log10(L)
	L2_err = []
	for n in n_list:
		u, data = gen_datasets(n,L, num_datasets, B)
		param_store, gv_store, gl = fit_functions(data, num_fits, start_point, end_point, logspace=True)
		L2_err.append(np.mean(np.sum((param_store - u[None, None, :])**2, -1).T, 1)/n**2)

	plt.clf()

	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')
	plt.loglog(gl,gl[:,None]*np.array(L2_err).T)
	plt.xlabel('Rankings')
	plt.ylabel('$(\ell/n^2)||\hat{u}_{MLE}-u^\star||_2^2$')
	plt.ylim(10**0, 10**2)
	plt.legend(list(map(lambda x: f"d = {x*(x-1)}", n_list)))
	plt.savefig('l2errCRSloglog_variousn.pdf',bbox_inches='tight')

def generate_CRS_errors_variousr(n_r_list=[(8,2), (8,4), (8,6), (8,8), (12,3), (12,6), (12,9), (12,12)], L=10000, num_datasets=20, B = 1.5, num_fits=20):
	start_point = np.log10(L/(100)); end_point = np.log10(L)
	for n, r in n_r_list:
		L2_err = []
		params = []
		u, data = gen_datasets(n,L, num_datasets, B, d=r)
		param_store, gv_store, gl = fit_functions(data, num_fits, start_point, end_point, d=r, logspace=True)
		params.append([param_store, gv_store, gl, u])
		L2_err.append(np.mean(np.sum((param_store - u[None, None, :])**2, -1).T, 1)/(n*r))

		filename = 'list'+''.join([f'_{n}_{r}'])+'.pkl'
		with open(filename, 'wb') as outfile:
			pkl.dump([L2_err, params], outfile)



def plot_CRS_errors_variousr(n_r_list=[(20,2), (20,4), (30,3), (30,6), (40,4), (40,8)]):
	L2_err = []
	plt.clf()
	for n,r in n_r_list:
		filename = 'list'+''.join([f'_{n}_{r}'])+'.pkl'
		with open(filename, 'rb') as infile:
			saved_values = pkl.load(infile)
			param_store, gv_store, gl, u = saved_values[1][0]
			L2_err.append(np.mean(np.sum((param_store - u[None, None, :])**2, -1).T, 1)/(n*r))
	
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')
	plt.loglog(gl,gl[:,None]*np.array(L2_err).T)
	plt.xlabel('Rankings')
	plt.ylabel('$(\ell/nr)||\hat{u}_{MLE}-u^\star||_2^2$')
	plt.ylim(3*10**0, 3*10**2)
	plt.legend(list(map(lambda x: f"n = {x[0]}, r = {x[1]}", n_r_list)))
	plt.savefig('l2errCRSloglog_variousn_r.pdf',bbox_inches='tight')
