import numpy as np
import torch,os,argparse,pickle
import torch.nn as nn
from scrape import *
import math
from functools import reduce
import cdm_pytorch as cp
from scipy.special import comb

def get_setsizes(X):
    set_sizes = X.sum(1)

    set_sizes_dict = {}
    for idx, val in enumerate(set_sizes):
        if val in set_sizes_dict:
            set_sizes_dict[val].append(idx)
        else:
            set_sizes_dict[val] = [idx]
    return set_sizes_dict
def evaluate_loss_setsizes(perms,n,model_list, Losses = None, batch_size=1):
    """
    returns losses given by the model for the specified loss criterion
    grouped by input setsize for some fold

    Args:
    perms- list of permutations
    n- number of items
    model_list- list of tuples of (choice model name, choice model)
    criterion- loss function used
    Losses- losses already computed
    """
    if sum(map(len,perms))==0:
        return Losses
    X,Y = RS_choices(perms,n)
    Y_X = np.zeros_like(X)
    Y_X[np.arange(len(Y)), Y] = 1
    
    set_sizes_dict = get_setsizes(X)

    for set_size, idxs in set_sizes_dict.items():
        for model in model_list:
            val_loss = [-item for item in cp.eval(X[idxs],Y_X[idxs], model[1], batch_size=batch_size)]
            Losses[model[0]][set_size] += val_loss

    return Losses



    # X = torch.Tensor(X); Y = torch.Tensor(Y)
    # dataset = torch.utils.data.TensorDataset(X,Y)
    # testloader = torch.utils.data.DataLoader(dataset)
    # dataiter = iter(testloader)
    # for data in testloader:
    #     x,y = data
    #     for model in model_list:
    #         output = model(Variable(x))
    #         target = Variable(y)
    #         loss = output.data.numpy()
    #         Losses[str(model)][int(torch.sum(x))].append(loss[0][int(y.numpy())])
    # return Losses

def evaluate_log_loss(perms,n,model_list,Losses = None,Losses_unif = None,RE=False):
    """
    returns losses given by the model for the specified loss criterion
    grouped by input setsize

    Args:
    perms- permutations
    n- number of alternatives
    model_list- list of tuples of (choice model name, choice model)
    Losses- losses already computed
    Losses_unif- unif losses already computed
    re- whether we're doing re
    """

    #unif_losses[k] is loss of uniform dist for top k list
    unif_losses = np.cumsum(list(map(np.log,range(1,n+1)[::-1])))
    # for model in model_list:
    #     ls[model[1]]=0
    sigma_lens = np.array([len(sigma) for sigma in perms],dtype=int)
    sigma_idx = np.repeat(np.arange(len(sigma_lens)),sigma_lens)
    X,Y = RS_choices(perms,n)
    Y_X = np.zeros_like(X)
    Y_X[np.arange(len(Y)), Y] = 1

    for model in model_list:
        val_loss = cp.eval(X,Y_X, model[1], batch_size=1)
        val_loss = np.bincount(sigma_idx, weights=val_loss)
        #if RE, must adjust probs for normalization, Z_k(RE,p) = (n choose k)
        val_loss = val_loss - comb(n, sigma_lens) if RE else val_loss
        Losses[model[0]] += list(val_loss)
        Losses_unif[model[0]] = list(unif_losses[sigma_lens-1]-val_loss)

    return Losses,Losses_unif



    # for sigma in perms:
    #     for model in model_list:
    #         ls[model[0]]=0
    #     if RE: #must adjust probs for normalization, Z_k(RE,p) = (n choose k)
    #         for i in range(1,len(sigma)):
    #             ls[model[0]]-=np.log(i)
    #         for i in range(n-len(sigma)+1,n+1):
    #             ls[model[0]]+=np.log(i)

    #     if not sigma: #when sigma is empty we skip
    #         continue

    #     for model in model_list:
    #         X,Y= RS_choices([sigma],n)#don't need flipped for RE-flipped at train time
    #         Y_X = np.zeros_like(X)
    #         Y_X[np.arange(len(Y)), Y] = 1

    #         for idx in range(len(sigma)):
    #             S = torch.Tensor(X[idx,:])
    #             S = Variable(S[None,:])
    #             ch = Y[idx].astype(int)

    #             loss = model[1](S).data.numpy()
    #             ls[model[0]] -= loss[0,ch]
    #             if math.isnan(loss[0,ch]):

    #                 Q=model[1].parameters().next().data.numpy()
    #                 s = S.data.numpy().astype(int)
    #                 s = np.array([x for x in range(n) if s[0,x]==1])
    #                 assert False

    #     for model in model_list:
    #         improvement = unif_losses[len(sigma)-1]-ls[model[0]]
    #         #assert not math.isnan(improvement)
    #         #assert not math.isnan(ls[str(model)])
    #         Losses_unif[model[0]].append(improvement)
    #         Losses[model[0]].append(ls[model[0]])

    # return Losses,Losses_unif

def max2(l,default=0):
    """
    returns the max of a list, or default if the list is empty
    helper function
    """
    if l ==[]:
        return default
    return max(l)

def trawl(path,dset,dtype,setsize,RE):
    """
    trawls over a directory and fits models to all data files
    """
    save_path = os.getcwd()+os.sep+'cache'+os.sep+'computed_errors'+os.sep+dset+os.sep
    Losses = {}; Losses_unif = {}

    for modelpath in os.listdir(path):
        if not modelpath.endswith('.p'):
            continue
        if RE != modelpath.endswith('RE.p'):
            continue

        cv_data = pickle.load(open(path+os.sep+modelpath,'rb'))
        models = [x for x in cv_data if x not in ['train','test','data']]
        L = cv_data['data']
        n = reduce(lambda x,y: max(x,y),map(max2,L))+1
        #if 'n' not in Losses:
        #    Losses['n']=n
        print ('Computing losses for '+modelpath)

        train = cv_data['train']; test = cv_data['test']; L = cv_data['data']
        Losses[modelpath]={}#{'n':n}
        Losses_unif[modelpath]={}
        assert modelpath in Losses
        if setsize:
            for model in models:
                Losses[modelpath][model]={}
                Losses_unif[modelpath][model]={}
                for i in range(n):
                    Losses[modelpath][model][i+int(setsize)]=[]
                    Losses_unif[modelpath][model][i+int(setsize)]=[]
        else:
            for model in models:
                Losses[modelpath][model]=[]
                Losses_unif[modelpath][model]=[]


        K = len(test)
        for k in range(K):
            test_perms  = [L[x] for x in test[k]]
            #for sigma in test_perms:
            #    print sigma
            model_list = [(model, cv_data[model][k]) for model in models]
            if setsize:
                assert not RE #can't do RE for partial lists at the moment
                Losses[modelpath] = evaluate_loss_setsizes(test_perms,n,model_list,Losses=Losses[modelpath])
            else:
                Losses[modelpath],Losses_unif[modelpath] = evaluate_log_loss(test_perms,n,model_list,Losses=Losses[modelpath],Losses_unif=Losses_unif[modelpath],RE=RE)


    s=''
    if RE:
        s+='-RE'
    if setsize:
        pickle.dump(Losses,open(save_path+dtype+'-setsize'+s+'.p','wb'))
    else:
        #for key in Losses:
            #for model in Losses[key]:
            #    if model == 'n':
            #        continue
            #    print key,model, np.mean(Losses[key][model]), np.mean(Losses_unif[key][model])
        #pickle.dump(Losses_unif,open(save_path+dtype+'-Lunif'+s+'.p','wb'))
        pickle.dump(Losses,open(save_path+dtype+'-Llog'+s+'.p','wb'))

if __name__ == '__main__':
    np.set_printoptions(suppress=True, precision=3)
    parser = argparse.ArgumentParser(description='ctr data parser')
    parser.add_argument('-dset', help="dataset name", default=None)
    parser.add_argument('-dtype', help="dataset type", default='soi')
    parser.add_argument('-setsize', help = 'whether to compute losses by setsize (y/n)', default='n')
    parser.add_argument('-re', help = 'whether to compute for RE models (y/n)', default='n')

    #parser.add_argument('-epochs', action="number of epochs to use", dest="c", type=int)

    args = parser.parse_args()

    if args.dset not in ['sushi','soi','nascar','letor','soc','election']:
        print('invalid dataset')
        assert False
    if args.dtype not in ['soi','soc']:
        print('invalid datatype')
        assert False
    if args.dset=='soc':
        args.dtype='soc'

    RE = args.re == 'y'
    setsize = args.setsize == 'y'
    path = os.getcwd()+os.sep+'cache'+os.sep+'learned_models'+os.sep+args.dset
    trawl(path,args.dset,args.dtype,setsize,RE)
