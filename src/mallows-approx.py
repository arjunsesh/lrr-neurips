import numpy as np
import argparse
import os
from random import shuffle
from scrape import *
#from sklearn.cross_validation import KFold
from sklearn.model_selection import KFold
from scipy.stats import sem
from scipy.special import gammaln
from scipy.optimize import minimize

def fit_mallows_approx(perms,n):
    """
    fits an approximation of the Mallows model:
    P(sigma; sigma_0, theta) prop to exp(theta*tau(sigma,sigma_0))

    Args:
    perms- list of partial rankings
    n- number of items ranked

    Returns:
    sigma_0- reference permutation
    theta- concentration parameter
    """
    C = np.zeros((n,n))
    pairs = 0
    length_counts = np.zeros(n)
    for sigma in perms:
        k = len(sigma)
        unranked = [x for x in range(n) if x not in sigma]
        pairs+= k*(k-1)/2
        pairs+= k*(n-k)
        for idx in range(k):
            C[sigma[idx],sigma[idx:]]+=1
            C[sigma[idx],unranked]+=1
        length_counts[k-1] += 1
    np.fill_diagonal(C,0)

    sigma_0 = []
    S = list(range(n))
    sigma_0_invs = 0
    for i in range(n):
        invs = np.sum(C[S,:][:,S],axis=0)
        next = S[np.argmin(invs)]
        sigma_0.append(next)
        sigma_0_invs+=np.amin(invs)#invs[S.index(next)]
        S.remove(next)
    # fixing this

    theta_old = np.log((float(sigma_0_invs)+1)/(float(pairs)+1))
    res =  minimize(neg_log_L_mallows_topK, .1, bounds = [(0, 100)], args = (length_counts,n,sigma_0_invs))
    theta = -res.x[0]
    print(theta,theta_old)
    #assert theta != -.1
    return sigma_0,theta


def neg_log_L_mallows_topK(theta,lengths,n,swaps):
    exps = np.exp(np.array(range(n))*-theta) #ith entry is e^-theta* i
    Z_S = np.cumsum(exps[::-1]) #j-th entry is Z_S for j-th choice in RS
    #print(Z_S)
    log_Z_S = np.log(Z_S)
    #print(log_Z_S)
    log_Z_k = np.cumsum(Z_S)
    log_Z_sum = np.dot(lengths,log_Z_k)
    #print(swaps)
    #print(log_Z_sum)
    return  swaps * theta + log_Z_sum

def inversions(sigma_0,sigma):
    """
    counts the number of pairs inverted (Kendall's tau distance) between two
    partial rankings

    Args:
    sigma_0- full ranking
    sigma- partial or full ranking
    """

    #restrict sigma_0 to items ranked by sigma
    sigma_0_S = [x for x in sigma_0 if x in sigma]

    #count up inversions
    for x in sigma_0_S:
        inv+=sigma.index(x)
    return inv

def mallows_choice_prob(S,sigma_0,theta):
    """
    computes choice probabilities P(.,S) under a Mallows
    model with reference perm sigma_0 and concentration param theta

    Args:
    S- choice set
    sigma_0- reference permutation for Mallows model
    theta- concentration param for Mallows model
    """
    sig = [x for x in sigma_0 if x in S]
    loc = list(map(lambda i: sig.index(i),S))
    p = np.exp(-theta* np.array(loc))
    return p/np.sum(p)

def log_RS_mallows_prob_partial(sigma_0,theta,sigma):
    """
    returns the log of probability of partial or full ranking under Mallows

    Args:
    sigma_0- reference perm
    theta- concentration parameter
    sigma- perm to compute probability of
    """
    n = len(sigma_0)
    log_p = 0
    S = list(range(n))
    for i in range(len(sigma)):
        probs = mallows_choice_prob(S,sigma_0,theta)
        log_p -= np.log(probs[S.index(sigma[i])])
        S.remove(sigma[i])
    return log_p

def log_n_choose_k(n,k):
    return gammaln(n+1)-gammaln(n-k+1)-gammaln(k+1)

def log_RE_mallows_prob_partial(sigma_0,theta,sigma):
    """
    returns the log of probability of partial or full ranking under Mallows

    Args:
    sigma_0- reference perm
    theta- concentration parameter
    sigma- perm to compute probability of
    """
    n = len(sigma_0)
    log_p = 0
    S = [i for i in sigma]
    for i in range(len(sigma)):
        probs = mallows_choice_prob(S,sigma_0,theta)
        log_p -= np.log(probs[S.index(sigma[i])])
        S.remove(sigma[i])
    return log_p+log_n_choose_k(n,len(sigma))

def test_mallows_approx_unif(sigma_0,theta,sigmas,re=False):
    """
    computes log-liklihood of mallows for test lists

    Args:
    sigma_0- reference permutation
    theta- concentration parameter
    sigmas- list of permutations to compute probability of
    """
    n = len(sigma_0)
    #compute log probs for test rankings

    if re:
        losses = map(lambda sigma: log_RE_mallows_prob_partial(sigma_0,theta,sigma),sigmas)
    else:
        losses = map(lambda sigma: log_RS_mallows_prob_partial(sigma_0,theta,sigma),sigmas)

    return np.mean(list(losses))

def cv(L,n,K=5,re=False):
    """
    trains and saves choosing to rank models with SGD via k-fold cv

    Args:
    L- list of data rankings
    n- number of items ranked
    model - choice models to fit
    K- number of folds
    re- whether doing re
    """

    kf = KFold(n_splits=K,shuffle=True)
    splits = kf.split(L)
    split_store_RS = {'train':[],'test':[],'data':L,'mallows':[],'L_log':[]}
    for k,(train,test) in enumerate(splits):
        print('fold'+str(k))
        if re:
            train_lists = [L[x][::-1] for x in train]
        else:
            train_lists = [L[x] for x in train]
        sigma_0_hat, theta_hat = fit_mallows_approx(train_lists,n)
        #print sigma_0_hat, theta_hat,re
        split_store_RS['mallows'].append({'sigma_0':sigma_0_hat,'theta':theta_hat})

        #store everything
        split_store_RS['train'].append(train)
        split_store_RS['test'].append(test)
        if re:
            test_lists = [L[x][::-1] for x in train]
        else:
            test_lists = [L[x] for x in test]
        split_store_RS['L_log'].append(test_mallows_approx_unif(sigma_0_hat,theta_hat,test_lists,re))
        k+=1

    return split_store_RS

def trawl(path,dset,dtype,cache=False,RE=True,check_trained=False,show_all = False):
    """
    trawls over the set of learned models for a given collection of datasets,
    and fits and tests the Mallows approximation over the cached test and train
    splits

    Args:
    path- directory containing pickled cv train/test splits
    dset- collection of datasets
    dtype- 'soi' or 'soc'
    cache- whether to cache the outputs
    RE- whether to do repeated eliminaton
    show_all- whether to print out each dataset individually
    """
    job_list = []
    save_path = os.getcwd()+os.sep+'cache'+os.sep+'learned_models'+os.sep+args.dset+os.sep
    files = os.listdir(path)
    shuffle(files)
    L_log_RS = []
    L_log_RE = []

    DATASETS = [fname[:-6]+'.'+dtype for fname in os.listdir(os.getcwd()+os.sep+'cache'+os.sep+'learned_models'+os.sep+dset)]
    for filename in files:
        if filename.endswith(dtype):

            #print filename
            filepath = path+os.sep+filename
            print(filename)
            if filename not in DATASETS and check_trained:
                continue
            if args.dtype=='soi':
                L,n = scrape_soi(filepath)
                #print n
                #print map(len,L)
            else:
                L,n = scrape_soc(filepath)
            print(n,len(L),sum(map(len,L)))
            if len(L)<10:
                continue
            if (dset == 'soi' or dset=='soc') and (n>50 or len(L)>1000):
                continue

            split_store_RS = cv(L,n,re=False)
            if cache:
                pickle.dump(split_store_RS,open(save_path+filename[:-4]+'-'+dtype+'-mallows.p','wb'))
            if show_all:
                print('RS L_log mean and sem for: '+filename)
                print(np.mean(split_store_RS['L_log']),sem(split_store_RS['L_log']))
            L_log_RS.extend(split_store_RS['L_log'])
            if RE:
                split_store_RS = cv(L,n,re=True)
                L_log_RE.extend(split_store_RS['L_log'])
                if cache:
                    pickle.dump(split_store_RS,open(save_path+filename[:-4]+'-'+dtype+'-mallows-RE.p','wb'))

    print('RS L_log mean and sem')
    L_log_RS = np.array(L_log_RS)
    print(np.mean(L_log_RS))
    print(sem(L_log_RS))
    if RE:
        print('RE L_log mean and sem')
        print(np.mean(L_log_RE))
        print(sem(L_log_RE))

if __name__ == '__main__':

    #code for running from the command line
    np.set_printoptions(suppress=True, precision=3)
    parser = argparse.ArgumentParser(description='mallows approx data parser')
    parser.add_argument('-dset', help="dataset name", default=None)
    parser.add_argument('-dtype', help="dataset type", default ='soi')
    parser.add_argument('-re', help="whether to do RE", default = 'n')
    parser.add_argument('-trained',help='whether to only compute mallows approx for dataset with already trained models in cache', default = 'n')
    parser.add_argument('-show_all', help = 'whether to print out the statistics for each dataset', default = 'n')
    args = parser.parse_args()
    re = (args.re == 'y')
    if args.dset not in ['sushi','soi','nascar','letor','soc','election']:
        print('dataset does not exist')
        assert False
    if args.dtype not in ['soi','soc']:
        assert False
    check_trained = (args.trained =='y')
    show_all = (args.show_all == 'y')
    path = os.getcwd()+os.sep+'data'+os.sep+args.dset
    #if args.dset == 'soi':
    #    path += os.sep+'filtered'
    trawl(path,args.dset,args.dtype,RE=re,check_trained=check_trained,show_all = show_all)
