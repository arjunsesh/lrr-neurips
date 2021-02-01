import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import sem
import os,argparse,pickle
from matplotlib import rc

def plot_one_scores_setsizes_with_hist(Scores,dset,dsetnum,dtype):
    """
    plots choice probability log losses vs choice setsize for a single dataset,
    along with a histogram showing the number of such choices, i.e. the
    partial ranking lengths for that dataset

    Args:
    Scores- dictionary of losses computed by test.py
    dset- name of dataset directory, e.g. election
    dsetnum- name of particular dataset considered, e.g. dublin-north
    dtype- '.soi' for partial rankings, '.soc' for full rankings
    """
    #compute number of alternaties
    n = max([k for k in Scores[list(Scores.keys())[0]].keys()])

    #boolean of whether we are considering a dataset primed for repeated elimination
    re = ('RE' in dsetnum)
    #helper string
    s = ''
    if re:
        s+= '-RE'

    plt.figure(figsize=(9,7))
    ax1 = plt.subplot2grid((3,1), (0,0), rowspan=2)
    ax2 = plt.subplot2grid((3,1), (2,0), rowspan=1)

    #compute losses and errorbars for all the choice models we may have considered
    for model in ['PL','CRS, r = 1','CRS, r = 4','CRS, r = 8','PCMC']:
        positions = [];means=[];sems=[];sizes=[]
        if model not in Scores:
            continue

        for i in Scores[model]:
            if len(Scores[model][i])==0:
                continue

            positions.append(n-i+1)
            scores = np.array(Scores[model][i])
            means.append(np.mean(scores))
            sems.append(sem(scores))

        if n-1 in positions and n not in positions:
            positions = [n]+positions
            means =  [0] + means
            sems = [0] + sems
        positions = np.array(positions);means=-np.array(means);sems=np.array(sems)
        ax1.errorbar(positions,means,yerr=sems,label=model,marker='x')

    #get name for saving plot
    dashes = [pos for pos, char in enumerate(dsetnum) if char == '-']
    last_dash = dashes[-1-int(re)]
    dset_name = dsetnum[:last_dash]

    #compute L_unif by adjusting for choice set size
    unif_losses = np.array(list(map(lambda pos: np.log(n-pos+1),positions)))
    if re:
        unif_losses = unif_losses[::-1]

    #make a pretty plot
    ax1.plot(positions,unif_losses,label='uniform',linestyle='--')
    ax1.set_xlim(.5,np.amax(positions)+.5)
    ax1.set_xticks(positions)
    ax1.set_xlabel('k (position in ranking)')
    ax1.set_ylim(0,ax1.get_ylim()[1])
    ax1.set_ylabel(r'$\ell(k;\hat \theta_{MLE},T)$')
    ax1.set_title(r'{\tt '+dset_name+s+r'}')
    ax1.legend(loc='best')

    #count how many times each position occured
    counts = np.zeros(n)
    m = Scores.keys()[0]
    for i in Scores[m]:
        pos = n-i+1
        counts[pos-1]+=len(Scores[m][i])
    counts[-1]=counts[-2]
    ax2.bar(range(1,n+1),counts,align='center')
    #get name for saving plot
    dashes = [pos for pos, char in enumerate(dsetnum) if char == '-']
    re = ('RE' in dsetnum)
    s = ''
    if re:
        s+= '-RE'
    last_dash = dashes[-1-int(re)]
    dset_name = dsetnum[:last_dash]
    ax2.set_xlabel('k (position in ranking)')
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(positions)
    if n<30:
        ax2.set_xticks(range(1,n+1))
    ax2.set_ylabel(r'\# rankings with'+'\n'+r'$\geq k$ positions')
    #ax2.set_title(r'{\tt '+dset_name+s+'}, ranking lengths')

    if dset=='nascar':
        ax1.set_xticks([x for x in positions if x==1 or x%5==0])
        ax2.set_xticks([x for x in positions if x==1 or x%5==0])
    #ax2.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.getcwd()+os.sep+'plots'+os.sep+dset+os.sep+dset_name+s+'-hist.pdf')
    plt.clf()

def plot_one_scores_setsizes(Scores,dset,dsetnum,dtype):
    """
    plots choice probability log losses vs choice setsize for a single dataset,
    but without the histogram

    Args:
    Scores- dictionary of losses computed by test.py
    dset- name of dataset directory, e.g. election
    dsetnum- name of particular dataset considered, e.g. dublin-north
    dtype- '.soi' for partial rankings, '.soc' for full rankings
    """
    n = max([k for k in Scores[list(Scores.keys())[0]].keys()])
    re = ('RE' in dsetnum)
    s = ''
    if re:
        s+= '-RE'

    for model in ['PL','CRS, r = 1','CRS, r = 4','CRS, r = 8']:
        positions = [];means=[];sems=[];sizes=[]
        if model not in Scores:
            continue

        for i in Scores[model]:
            if len(Scores[model][i])==0:
                continue

            positions.append(n-i+1)
            #sizes.append(i)
            scores = np.array(Scores[model][i])
            means.append(np.mean(scores))
            sems.append(sem(scores))

        if n-1 in positions and n not in positions:
            positions = [n]+positions
            means =  [0] + means
            sems = [0] + sems

        positions = np.array(positions);means=-np.array(means);sems=np.array(sems)
        if re:
            positions = positions[::-1]

        plt.errorbar(positions,means,yerr=sems,label=model,marker='x')

    #get name for saving plot
    dashes = [pos for pos, char in enumerate(dsetnum) if char == '-']

    last_dash = dashes[-1-int(re)]
    dset_name = dsetnum[:last_dash]
    unif_losses = np.array(list(map(lambda pos: np.log(n-pos+1),positions)))
    if re:
        unif_losses = unif_losses[::-1]
    plt.plot(positions,unif_losses,label='uniform',linestyle='--')
    plt.xlim(.9,np.amax(positions)+.1)
    plt.xlabel('k (position in ranking)')
    plt.xticks(positions)
    plt.ylabel(r'$\ell(k;\hat \theta_{MLE},T)$')
    plt.title(r'{\tt '+dset_name+s+r'}')#, $\ell_{log}(\cdot,\hat \theta_{MLE})$ vs. position')
    plt.legend(loc='best')
    plt.tight_layout()



    plt.savefig(os.getcwd()+os.sep+'plots'+os.sep+dset+os.sep+dset_name+s+'.pdf')
    plt.clf()

def print_one_losses(Scores,dset,dsetnum,dtype,unif=False,re=False):
    """
    outputs the log losses for one dataset to a text file

    Args:
    Scores- log losses as a function of choice set size
    dset- collection of datasets this dataset belongs to
    dsetnum- specific name of dataset among the collection
    dtype- 'soi' or 'soc'
    unif- whether to output L_unif (see paper) or standard log loss
    re- whether this was a repeated elimination model
    """
    means = []; sems = [];labels = []
    model_list = ['PL','CRS, r = 1','CRS, r = 4','CRS, r = 8','PCMC']
    for model in model_list:
        if model not in Scores:
            continue
        labels.append(model)
        scrs = np.array(Scores[model])
        means.append(np.mean(scrs))
        sems.append(sem(scrs))

    means = np.array(means)
    sems = np.array(sems)
    if unif:
        s = 'unif'
    else:
        s = 'log'
    if re:
        s+= '-'

    with open(os.getcwd()+os.sep+'plots'+os.sep+dset+os.sep+dsetnum+'-'+dtype+'-L'+s+'.txt','w') as f:
        f.write('models:')
        for idx in range(len(labels)):
            model = labels[idx]
            f.write(model + ',')
        f.write('\nlosses:')
        for idx in range(len(labels)):
            log_loss = means[idx]
            f.write(("%.3f" % log_loss)+' & ')
        f.write('\nse:')
        for idx in range(len(labels)):
            se = sems[idx]
            f.write(("%.3f" % se)+ ' & ')

def print_all_losses(Scores,dset,dtype,unif=False,re=False):
    """
    outputs the log losses for one dataset to a text file

    Args:
    Scores- log losses as a function of choice set size
    dset- collection of datasets this dataset belongs to
    dsetnum- specific name of dataset among the collection
    dtype- 'soi' or 'soc'
    unif- whether to output L_unif (see paper) or standard log loss
    re- whether this was a repeated elimination model
    """
    means = {}; sems = {}
    model_list = ['PL','CRS, r = 1','CRS, r = 4','CRS, r = 8']
    rankings = 0
    for dsetid in Scores:
        for model in model_list:
            if model not in means:
                means[model]=[]
                sems[model]=[]

            if model in Scores[dsetid]:
                scrs = np.array(Scores[dsetid][model])
                rankings += int(model=='PL')*len(scrs)
                means[model].append(np.mean(scrs))
                sems[model].append(sem(scrs))
            elif model=='CRS, r = 8' and 'CRS, r = 4' in Scores[dsetid]:
                scrs = np.array(Scores[dsetid]['CRS, r = 4'])
                means[model].append(np.mean(scrs))
                sems[model].append(sem(scrs))
            else:
                scrs = np.array(Scores[dsetid]['CRS, r = 1'])
                means[model].append(np.mean(scrs))
                sems[model].append(sem(scrs))

    means_list = []
    sems_list = []
    labels = []
    print('datasets, rankings:')
    print(len(means['PL']),rankings)
    for model in model_list:
        if model not in means:
            continue
        labels.append(str(model))
        means_list.append(np.mean(means[model]))
        sems_list.append(np.mean(sems[model]))
    means = np.array(means_list)
    sems = np.array(sems_list)

    if unif:
        s = 'unif'
    else:
        s = 'log'
    if re:
        s += '-RE'
    with open(os.getcwd()+os.sep+'plots'+os.sep+dset+os.sep+dtype+'-L'+s+'-all.txt','w') as f:
        f.write('models:')
        for idx in range(len(labels)):
            model = labels[idx]
            f.write(model + ' & ')
        f.write('\nlosses:')
        for idx in range(len(labels)):
            log_loss = means[idx]
            f.write(("%.3f" % log_loss)+' & ')
        f.write('\nse:')
        for idx in range(len(labels)):
            se = sems[idx]
            f.write(("%.3f" % se)+' & ')

def parse():
    """
    Handles command line inputs and outputs correct plots or statistics
    """

    #set some parameters to make plots prettier
    np.set_printoptions(suppress=True, precision=3)
    plt.rcParams.update({'font.size': 14})
    rc('text', usetex=True)

    #argparser reads in the plots/data we want
    parser = argparse.ArgumentParser(description='ctr data parser')
    parser.add_argument('-dset', help="dataset name", default=None)
    parser.add_argument('-dtype', help="data type", default='soi')
    parser.add_argument('-setsize', help = 'whether to compute losses by setsize', default='n')
    parser.add_argument('-all', help='whether to aggregate over all datasets in directory (y/n)', default='y')
    parser.add_argument('-re', help='whether to plot for RE models (y/n)', default='n')
    parser.add_argument('-hist', help='whether to include a histogram of the ranking lengths(y/n)', default='n')
    args = parser.parse_args()

    #checks whether the dataset is in the right place
    if args.dset not in os.listdir(os.getcwd()+os.sep+'cache'+os.sep+'computed_errors'):
        print('no errors found in cache to plot')
        assert False

    #checks whether the datatype is known
    if args.dtype not in ['soi','soc']:
        print('invalid datatype')
        assert False

    #compute booleans and/or strings based on other input arguments
    all = (args.all == 'y')
    setsize = (args.setsize=='y')
    re = (args.re=='y')
    hist = (args.hist=='y')
    s=''
    if re:
        s+='-RE'

    #compute filepath of errors to plot
    path = os.getcwd()+os.sep+'cache'+os.sep+'computed_errors'+os.sep+args.dset+os.sep

    #whether we are grouping losses by choice set size
    if setsize:
        Scores = pickle.load(open(path+args.dtype+'-setsize'+s+'.p', 'rb'))
    else:
        Scores = pickle.load(open(path+args.dtype+'-Llog'+s+'.p','rb'))

    #call the appropriate plotting or printing function
    if all:
        if setsize:
            print('comparing losses as a function of choice set size across different datasets is not supported')
        else:
            #outputs combined losses for all the datasets in the folder to a text file
            print('computing losses for all datasets in '+args.dset)
            print_all_losses(Scores,args.dset,args.dtype,re=re)
    elif setsize:
        for dataset in Scores:
            print('plotting losses for '+dataset)
            if args.dtype == 'soi' and hist:
                plot_one_scores_setsizes_with_hist(Scores[dataset],args.dset,dataset,args.dtype)
            else:
                plot_one_scores_setsizes(Scores[dataset],args.dset,dataset,args.dtype)

    else: #
        for dataset in Scores:
            print(dataset)
            print_one_losses(Scores[dataset],args.dset,dataset,args.dtype,re=re)

if __name__ == '__main__':
    parse()
