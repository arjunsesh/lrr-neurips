import networkx as nx
import numpy as np
import pickle

def scrape(file):
    """ scrapes rankings, counts from agg.txt file"""
    D={}
    G={}
    with open(file,'r') as f:
        for line in f:
            L = line.split(' ')
            qid = L[1][4:]
            if qid not in D:
                D[qid]=[]
                G[qid]=[]

            #ground truth
            G[qid].append(int(L[0]))
            #extract ranks
            ranks=[]
            for i in range(2,27):
                [l,rank]=L[i].split(':')
                if rank != 'NULL':
                    ranks.append(int(rank))
                else:
                    ranks.append(0)
            D[qid].append(ranks)


    C={};N={}
    for qid in D:
        C[qid]=[]
        N[qid] = len(D[qid])
        A= np.array(D[qid])
        assert A.shape[1] == 25
        for i in range(25):
            l = A[:,i]
            ranked = np.where(l>0)[0]
            ranking = ranked[np.argsort(l[ranked])]
            C[qid].append(ranking)
    #pickle.dump(C,open('MQ-lists.p','wb'))
    return C,N,G


def strongly_connected_restriction(lists,N):
    """restricts D's qid lists to a strongly connected component of the
    comparison hypergraph"""
    L = lists
    n = N
    A = np.zeros((n,n))
    for l in L:
        for idx in range(len(l)):
            x=l[idx]
            for y in l[idx:]:
                A[y,x]=1

    G = nx.DiGraph(data=A)
    comp = nx.strongly_connected_components(G)
    cnt =0
    gcc = []
    for c in comp:
        if len(c)>len(gcc):
            gcc = c

    if len(gcc)<n:
        print 'shrunk:'+qid+' from '+str(n)+' to '+str(len(gcc))
    gcc = list(gcc)
    for idx in range(len(L)):
        L[idx] = [gcc.index(x) for x in L[idx] if x in gcc]
    return L,range(len(gcc))

def list_str(l):
    s = '1,'
    for x in l:
        s+=str(x)
        s+=','
    return s[:-1]

def write_soi(qid,L,gcc):
    with open(qid+'.soi','w') as f:
        f.write(str(len(gcc))+'\n')
        for x in gcc:
            f.write(str(x)+'\n')
        f.write('begin data:\n')
        for l in L:
            if len(l)>0:
                f.write(list_str(l)+'\n')

if __name__ == '__main__':
    np.set_printoptions(suppress=True, precision=3)
    C,N,_ = scrape('agg.txt')
    for qid in C:
        print qid
        L,gcc = strongly_connected_restriction(C[qid],N[qid])
        write_soi(qid,L,gcc)
