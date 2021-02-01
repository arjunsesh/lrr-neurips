import sys,os,pickle,argparse
import numpy as np
from models import *
from sklearn.model_selection import KFold
#deprecated
#from sklearn.cross_validation import KFold
from scrape import *
import argparse
from torch.multiprocessing import Pool
from random import shuffle

import cdm_pytorch as cp

def fit(X,Y,model, epochs=20,batch_size=1,verbose=True,print_batches=1000,opt='Adam'):
    """
    Fits a choice model with pytorch's SGD

    X- Indicator vectors for choice sets
    Y- indices of choices
    model- tuple of (name of model, choice model to fit, parameters of the model)
    criterion- which loss function to use (default to negative log likelihood for MLE)
    epochs- number of times to loop over the training data
    batch_size- how large to make batches
    verbose- whether to print updates as training goes on
    print_batches- how often to print updates on training
    opt- which optimizer to use 'SGD' or 'Adam'
    """
    Y_X = np.zeros_like(X)
    Y_X[np.arange(len(Y)), Y] = 1
    model, tr_loss, gv = cp.l2err_run(X, Y_X, batch_size=batch_size, epochs=epochs, 
        Model=model[1], seed=2, lr=5e-3, verbose=True, **model[2])

    return model

def cv(L,n,models,save_path,K=5,epochs=20,batch_size=1,opt='Adam',seed=True,RE=False):
    """
    trains and saves choosing to rank models with SGD via k-fold cv

    Args:
    L- list of data rankings
    n- number of items ranked
    models - list of tuples of (name of model, choice model to fit, parameters of the model)
    save_path- folder to save to
    K- number of folds
    epochs- number of times to loop over the data
    """
    kf = KFold(n_splits=K,shuffle=True)
    splits = kf.split(L)
    split_store = {'train':[],'test':[],'data':L}
    for model in models:
        split_store[model[0]]=[]
    for k,(train,test) in enumerate(splits):


        print('Beginning fold '+str(k)+' of '+str(K))

        #scrape training choices and fit model
        X_train,Y_train = RS_choices([L[x] for x in train],n)
        for model in models:
            print('training '+model[0])
            if seed and model[0] == 'PCMC':
                utils = models[0][1].parameters().next().data.numpy()
                #print utils
                g= np.exp(utils)
                g/= np.sum(g)
                model = cp.PCMC(n,gamma=g)
            model_trained = fit(X_train,Y_train,model, epochs=epochs,batch_size=batch_size,opt=opt)
            split_store[model[0]].append(model_trained)

        #store everything
        split_store['train'].append(train)
        split_store['test'].append(test)

    if not RE:
        pickle.dump(split_store,open(save_path+'.p','wb'))
    else:
        pickle.dump(split_store,open(save_path+'-RE.p','wb'))
    return 0

def parallel_helper(tup):
    """
    unpacks a tuple so that we can apply the function cv in parallel
    (Pool does not allow mapping of an anonymous function)
    """
    L,n,models,save_path,epochs,batch_size,opt,seed,RE,K = tup
    return cv(L,n,models,save_path,epochs=epochs,batch_size=batch_size,opt=opt,seed=seed,RE=RE,K=K)

def ensure_dir(file_path):
    """
    helper function from stack overflow that automatically makes directories
    in the cache for me
    thanks to this:
    https://stackoverflow.com/questions/273192/how-can-i-safely-create-a-nested-directory-in-python
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def trawl(dset,dtype,epochs,parallel=False,batch_size=1,max_n=30,max_rankings=1000,opt='Adam',num_dsets=10,seed=True,RE=False,K=5):
    """
    trawls over a directory and fits models to all data files

    Args:
    dset- name of dataset(s) considered
    dtype- 'soi' for partial rankings, 'soc' for complete rankings
    epochs- number of times to loop over the data
    parallel- whether to train models in parallel over the datasets in the directory
    batch_size- number of choices to train on at a time
    max_n- largest number of alternatives allowed to train on a dataset
    max_rankings- maximum number of rankings to fit a dataset
    opt- which optimizer to use
    num_dsets- number of datasets to fit
    seed- whether to seed PCMC
    RE- whether to compute repeated elimianation (RS if false)
    K- number of CV folds for each dataset
    """
    #we will loop over the datasets stored in this directory
    path = os.getcwd()+os.sep+'data'+os.sep+dset
    files = os.listdir(path)
    #shuffle(files)

    #this is where we'll save the output models
    save_path = os.getcwd()+os.sep+'cache'+os.sep+'learned_models'+os.sep+dset+os.sep

    job_list = []
    batch = (batch_size>1)
    for filename in files:#loop over the directory
        print(filename)
        if filename.endswith(dtype):#will
            filepath = path+os.sep+filename
            if dtype=='soi':
                L,n = scrape_soi(filepath)
            else:
                L,n = scrape_soc(filepath)
            if len(L)<=10 or len(L)>max_rankings or n>max_n:
                if len(L)<=10:
                    reason = 'too few rankings- '+str(len(L))+', min is 10'
                elif len(L)>max_rankings:
                    reason = 'too many rankings- '+str(len(L))+', max is '+str(max_rankings)
                else:
                    reason = 'too many alternatives- '+str(n)+', max is '+str(max_n)
                print(filename+' skipped, '+reason)
                continue
            else:
                print(filename+' added')

            #collect models
            models = []
            for d in [1,4,8]:
                if d>n:
                    continue
                models.append((f'CRS, r = {d}', cp.CDM, {'embedding_dim': d}))
            models.append((f'PL', cp.MNL, {}))
            #models.append(PCMC(n,batch=batch))
            #models.append((f'RS-PCMC', cp.PCMC, {'batch': batch}))
            

            #append tuple containing all the ojects needed to train the model on the dataset
            job_list.append((L,n,models,save_path+filename[:-4]+'-'+dtype,epochs,batch_size,opt,seed,False,K))
            if RE:
                job_list.append((map(lambda x:x[::-1],L),n,models,save_path+filename[:-4]+'-'+dtype,epochs,batch_size,opt,seed,True,K))
            if len(job_list)>=num_dsets:
                print('maximum number of datasets reached')
                continue

    print(str(len(job_list))+' datasets total')
    print(str(sum(map(lambda x: len(x[0]),job_list)))+ ' total rankings')
    #sorts the jobs by number of alternatives*number of (partial) rankings
    #will roughly be the number of choices, up to partial ranking length
    sorted(job_list,key=lambda x: x[1]*len(x[0]))
    #training for each dataset can be done in parallel with this
    if parallel:
        p = Pool(4)
        p.map(parallel_helper,job_list)
    else:
        [x for x in map(parallel_helper,job_list)]

def parse():
    """
    parses command line args, run when train.py is __main__
    """
    np.set_printoptions(suppress=True, precision=3)
    parser = argparse.ArgumentParser(description='ctr data parser')
    parser.add_argument('-dset', help="dataset name", default=None)
    parser.add_argument('-dtype', help="dataset type", default ='soi')
    parser.add_argument('-epochs', help="number of epochs to use", default='100')
    parser.add_argument('-batch_size', help='batch_size for training', default = '1000')
    parser.add_argument('-max_n', help='maximum number of items ranked', default = '10')
    parser.add_argument('-max_rankings', help='maximum number of rankings', default = '1000')
    parser.add_argument('-opt', help='SGD or Adam', default='Adam')
    parser.add_argument('-num_dsets', help='how many datasets to use', default='100')
    parser.add_argument('-seed_pcmc', help='whether to seed pcmc with MNL (y/n)', default = 'n')
    parser.add_argument('-re', help='whether to train RE models (y/n)', default = 'n')
    parser.add_argument('-folds', help='number of folds for cv on each dataset', default='5')
    args = parser.parse_args()

    if args.dtype not in ['soi','soc']:
        print('wrong data type')
        assert False
    if args.opt not in ['SGD','Adam']:
        print('optmizer can be SGD or Adam')
        assert False
    if args.dset=='soc':
        args.dtype='soc'
    path = os.getcwd()+os.sep+'data'+os.sep+args.dset
    if args.dset == 'soi':
        path += os.sep+'filtered'
    if args.seed_pcmc not in ['y','n']:
        print('y or n required for -seed_pcmc')
    seed = (args.seed_pcmc=='y')
    RE = (args.re == 'y')
    K = int(args.folds)
    trawl(args.dset,args.dtype,epochs=int(args.epochs),batch_size=int(args.batch_size),
          max_n=int(args.max_n),max_rankings=int(args.max_rankings),opt=args.opt,
          num_dsets=int(args.num_dsets),seed=seed,RE=RE,K=K)

if __name__ == '__main__':
    parse()
