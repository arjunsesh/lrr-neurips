import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import itertools as it
from scipy.stats import kendalltau as kt
from scipy.special import softmax
import seaborn as sns

import cdm_pytorch as cp
from itertools import permutations
from functools import reduce
from scipy.optimize import minimize
import os
import matplotlib.colors as colors
from matplotlib.ticker import LogFormatterExponent # <-- one new import here
from matplotlib import rc
from mpl_toolkits.axes_grid1 import make_axes_locatable

def factorial(n):
	if n==1:
		return n
	else:
		return n*factorial(n-1)

def flip(i,L):
	L_ = list(L)
	L_[i] = L[i+1]
	L_[i+1] = L[i]
	return tuple(L_)

def plain_graph(seed = 1985, n =4):
	perms = list(it.permutations(range(n)))
	perms_shifted = list(it.permutations(range(1,n+1)))
	#perms_shifted_lookup = {k:v for (v,k) in enumerate(perms_shifted)}
	adj = {}
	for idx, node_1 in enumerate(perms):
		#adj_idx = np.argsort([kt(node_1, node_2).correlation for node_2 in perms])[::-1][1:n]
		sigma = perms_shifted[idx]
		perm_string = ''.join(map(str, sigma))
		neighbor_perms = [flip(idx,sigma) for idx in range(n-1)]
		adj[perm_string] = [''.join(map(str, sigma_neighbor)) for sigma_neighbor in neighbor_perms]
	G = nx.Graph(adj)
	pos2d = nx.spring_layout(G, dim=2, seed = seed)
	#pos2d = {k: v[[0,2]] for k,v in pos.items()}

	#plt.clf()
	#nx.draw(G, pos=pos2d, with_labels=True)

	return G, pos2d


def mnl_prob(theta, perm):
	perm = list(perm)
	return np.prod([softmax(theta[perm[idx:]])[0] for idx, item in enumerate(perm)])

def cdm_prob(U, perm):
	perm=list(perm)
	return np.prod([softmax(U[:, perm[idx:]].sum(1)[perm[idx:]])[0] for idx, item in enumerate(perm)])

def add_cdm_weights(U=-np.array([.25,.75,.5,1])[:,None].dot(np.array([4,3,2,1])[None,:])):
	U = np.copy(U)
	np.fill_diagonal(U,0)
	#assert U.shape == (4,4)
	perms = list(it.permutations(range(len(U))))
	weights = [cdm_prob(U, perm) for perm in perms]

	return weights

def add_mnl_weights(theta=[4,3,2,1]):
	theta = np.array(theta).flatten()
	perms = list(it.permutations(range(len(theta))))
	weights = [mnl_prob(theta, perm) for perm in perms]

	return weights

def one_ranking_to_choices(sigma,n=4):
	O = np.zeros((n-1,n))
	J = np.zeros((n-1,n))
	S = list(range(n))
	for idx,i in enumerate(sigma[:-1]):

		J[idx,i]=1
		for j in S:
			O[idx,j] = 1
		S.remove(i)
	return O,J

def many_rankings_to_choices(sigmas,n=4):
	OsJs = map(lambda sigma: one_ranking_to_choices(sigma,n=n),sigmas)
	return reduce(lambda OJ1,OJ2: (np.concatenate((OJ1[0],OJ2[0])),np.concatenate((OJ1[1],OJ2[1]))), OsJs)

def kt_swaps(sigma,sigma_0):
	swaps = 0
	for idx, i in enumerate(sigma):
		after = sigma[idx:]
		before_sigma_0 = sigma_0[:sigma_0.index(i)]
		swaps += len([x for x in before_sigma_0 if x in after])
	return swaps

def mallows_mixture_Sn(theta = 1.0,n=4, reversals=True, weights = [.5,.5]):
	S_n = [sigma for sigma in permutations(range(n))]
	P_1_to_n= []
	assert len(weights)==2
	assert weights[0]+weights[1]==1
	if reversals:
		other_center = range(n)[::-1]
	else:
		other_center = [2,1,3,0]
	#print(other_center)
	P_other = []
	for sigma in S_n:
		#print(sigma,kt_swaps(sigma,range(n)),kt_swaps(sigma,other_center))
		P_1_to_n.append(np.exp(-theta*kt_swaps(sigma,range(n))))
		P_other.append(np.exp(-theta*kt_swaps(sigma,other_center)))
	P_1_to_n = np.array(P_1_to_n)/np.sum(np.array(P_1_to_n))
	P_other = np.array(P_other)/np.sum(np.array(P_other))
	#print([x for x in zip(map(lambda sigma: ''.join(map(lambda x: str(x+1),sigma)),S_n),P_1_to_n,P_other)])

	return S_n,weights[0]*P_1_to_n+weights[1]*P_other

def fit_mallows(sigmas, n=4):
	S_n = [sigma for sigma in permutations(range(n))]
	best_sigma = None
	fewest_swaps = 99999999
	swap_cache = np.zeros((n,n))
	for data in sigmas:
		for idx,i in enumerate(data[:-1]):
			swap_cache[data[idx+1:],i] += 1
	for sigma in S_n:
		swaps = np.sum([np.sum(swap_cache[i,sigma[idx:]]) for (i,idx) in enumerate(sigma)])
		#reduce(lambda x,y: x+y, map(lambda tau: kt_swaps(sigma,tau),sigmas))
		#print(sigma,swaps)
		if swaps < fewest_swaps:
			best_sigma = sigma
			fewest_swaps = swaps

	m =  len(sigmas) #potential for n choose 2 flips per ranking
	res = minimize(neg_log_L_mallows, 1, bounds = [(0, 100)], args = (m,n,fewest_swaps))
	best_theta = res.x
	#\ell(theta; \sigma_0, D) = \sum_\sigma -\theta * kt(\sigma,\sigma_0) - \sum_k=1^n log(\sum_j=1^k exp(-theta*i))
	#second term doesn't depend on theta- still have to solve with software
	frac_swaps = fewest_swaps / (m * n * (n-1) / 2.0)
	#print(frac_swaps,best_theta)
	return best_sigma, best_theta

def neg_log_L_mallows(theta,m,n,swaps):
	exps = np.exp(np.array(range(n))*-theta)
	log_Z = 0
	for k in range(1,n):
		log_Z += np.log(np.sum(exps[:k+1]))
	return  swaps * theta + m * log_Z

def neg_log_L_mallows_prime(theta, m,n,swaps):
	one_to_n = np.array(range(n))
	exps = np.exp(-theta * one_to_n)
	exps_prime = np.multiple(exps,one_to_n)
	log_Z_prime = 0
	for k in range(1,n):
		log_Z_prime += np.sum(exps_prime[:k+1])/np.sum(exps[:k+1])
	return swaps + m * log_Z_prime

def add_mallows_weights(sigma_0,theta,n=4):
	S_n = [sigma for sigma in permutations(range(n))]
	p = list(map(lambda sigma: np.exp(-theta*kt_swaps(sigma,sigma_0)), S_n))
	return p/np.sum(p)

def plot_weights_better(ax,weights,pos,name,cmap, vmin, vmax, node_size = 1000, with_labels = True,alpha = .8):
	ax.axis("off")
	nx.draw_networkx(G,
		pos=pos2d,
		node_color=np.log(np.array(weights)),
		with_labels=with_labels,
		node_size = node_size,
		ax = ax,
		font_color = 'w',
		font_size = 11,
		cmap = cmap,
		vmin = vmin,
		vmax = vmax,
		alpha = alpha,
	)


if __name__ == '__main__':

	### PARAMETERS ###
	seed = 68952 # used in paper, uncomment line below for general figure
	# seed = np.random.randint(100000)
	theta_mallows_mixture = 1.0
	reversals = True
	weights = [.75,.25]
	n = 5
	alpha = .8
	m = 5000 # used in paper, 1000 also works well
	if n>4:
		m*=10

	#print(factorial(4))
	### COMPUTE ALL THE NODE WEIGHTS ###
	G,pos2d = plain_graph(seed, n=n)
	print('graph done')
	path = 'n_'+str(n)+'theta_'+str(theta_mallows_mixture)+'_m_'+str(m)+'_seed_'+str(seed)
	if not os.path.exists(path):
		os.mkdir(path)
	S_n,P_mallows_mix = mallows_mixture_Sn(theta_mallows_mixture,n=n, reversals = reversals, weights=weights)
	#i was goofin
	#P_mallows_mix = np.random.rand(24)
	#P_mallows_mix /= np.sum(P_mallows_mix)
	print('true dist computed')
	mallows_sample_idxs = np.random.choice(range(factorial(n)),size = m, p = P_mallows_mix)
	print('sampling done')
	mallows_sample = [S_n[idx] for idx in mallows_sample_idxs]
	empirical_distribution = (10e-4)*np.ones(factorial(n))
	for idx in mallows_sample_idxs:
		empirical_distribution[idx]+=1
	empirical_distribution/=np.sum(empirical_distribution)
	O,J = many_rankings_to_choices(mallows_sample,n=n)

	model, tr_loss, gv = cp.l2err_run(O,J, epochs=500, lr=5e-2, Model=cp.FullCDM)
	U_hat = model.U.weight.detach().numpy()[:-1,:-1].T.astype(np.float64)

	cdm_w_hat = add_cdm_weights(U = U_hat)
	print('fitting mallows')
	sigma_hat,theta_hat = fit_mallows(mallows_sample,n=n)
	print('mallows fit')
	mallows_w_hat = add_mallows_weights(sigma_hat,theta_hat,n=n)[:,0]

	model, tr_loss, gv = cp.l2err_run(O,J, epochs=500, lr=1e-2, Model=cp.MNL)
	utils_hat = model.logits.weight.detach().numpy()[:-1,:].astype(np.float64)

	luce_w_hat = add_mnl_weights(utils_hat)

	### PLOT CODE ###
	fig,axarr = plt.subplots(1,4, figsize = (24,5))
	cmap = 'viridis'
	#cmap = 'Blues'
	cmap_scalar = 1
	### PLOT THE COLORBAR ###
	vmin = cmap_scalar*np.log(np.min(np.array(list(map(lambda w: np.min(np.array(w)), [cdm_w_hat,luce_w_hat,mallows_w_hat,empirical_distribution,P_mallows_mix])))))
	vmax = np.log(np.max(np.array(list(map(lambda w: np.max(np.array(w)), [cdm_w_hat,luce_w_hat,mallows_w_hat,empirical_distribution,P_mallows_mix])))))
	print(vmin,vmax)
	np.random.seed(seed)
	#none of these params helped anything imo
	delta = .5
	posdict = {k: (delta*(2*np.random.random()-1),delta*(2*np.random.random()-1)) for k in G.nodes()}
	xoff = 1
	yoff = 0
	lambduh = 1
	one_to_n_str = ''.join(map(str,range(1,n+1)))
	posdict[one_to_n_str] = (-xoff*lambduh,-yoff*lambduh)
	posdict[one_to_n_str[::-1]] = (xoff*lambduh,yoff*lambduh)
	pos2d = nx.spring_layout(G, dim=2, fixed=list([one_to_n_str, one_to_n_str[::-1]]), pos=posdict, iterations = 1000, threshold = 1e-6)

	with_labels = (n<=4)
	node_size = 1000
	if n == 5:
		node_size = 600
	if n == 6:
		node_size = 100
	if n == 7:
		node_size = 50
	plot_weights_better(axarr[3],cdm_w_hat, pos2d,'cdm', cmap, vmin, vmax, node_size = node_size, with_labels = with_labels, alpha = alpha)
	plot_weights_better(axarr[1],luce_w_hat,pos2d, 'luce', cmap, vmin, vmax, node_size = node_size, with_labels = with_labels, alpha = alpha)
	plot_weights_better(axarr[2],mallows_w_hat, pos2d,'mallows', cmap, vmin, vmax, node_size = node_size, with_labels = with_labels, alpha = alpha)
	#plot_weights_better(axarr[0],empirical_distribution, pos2d,'empirical', cmap, vmin, vmax, node_size = node_size, with_labels = with_labels, alpha = alpha)
	plot_weights_better(axarr[0],P_mallows_mix, pos2d,'true', cmap, vmin, vmax, node_size = node_size, with_labels = with_labels, alpha = alpha)

	plt.tight_layout()
	plt.savefig(path+os.sep+'dists.pdf')
	plt.clf()
	plt.figure()
	sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
	sm.set_array([])# no clue why
	rc('text', usetex=True)
	cb = plt.colorbar(sm)
	ax = plt.gca()
	ax.axis("off")
	cb.set_label(r'$\log\ p(\sigma)$')
	#dunno how to do this cb.set_ticklabels([r'$e^{-{0}}$'.format(x) for x in cb.get_ticklabels()])
	plt.legend()
	plt.tight_layout()
	plt.savefig(path+os.sep+'colorbar.pdf')

	### OLD PLOT CODE ###
	#plt.clf(); sns.color_palette('pastel'); nx.draw(G, pos=pos2d, node_color=np.log(np.array(mallows_w_hat)), with_labels=True); plt.savefig('single_mallows_weight_graph_theta_is_'+str(theta_mallows_mixture)+'_m_is_'+str(m)+'_seed_is'+str(seed)+'.pdf')
	#plt.clf(); sns.color_palette('pastel'); nx.draw(G, pos=pos2d, node_color=np.log(np.array(luce_w_hat)), with_labels=True); plt.savefig('plackett_luce_graph_theta_is_'+str(theta_mallows_mixture)+'_m_is_'+str(m)+'_seed_is'+str(seed)+'.pdf')
	#plt.clf(); sns.color_palette('pastel'); nx.draw(G, pos=pos2d, node_color=np.log(np.array(empirical_distribution)), with_labels=True); plt.savefig('empirical_distribution_theta_is_'+str(theta_mallows_mixture)+'_m_is_'+str(m)+'_seed_is'+str(seed)+'.pdf')
	#plt.clf(); sns.color_palette('pastel'); nx.draw(G, pos=pos2d, node_color=np.log(np.array(P_mallows)), with_labels=True); plt.savefig('true_distribution_theta_is_'+str(theta_mallows_mixture)+'_m_is_'+str(m)+'_seed_is'+str(seed)+'.pdf')

	#
	#
	#
	#
