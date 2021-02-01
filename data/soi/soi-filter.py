import networkx as nx
import numpy as np
import pickle
import os

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
        print 'shrunk from '+str(n)+' to '+str(len(gcc))
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

def write_soi(filepath,L,gcc):
    with open(filepath,'w') as f:
        f.write(str(len(gcc))+'\n')
        for x in gcc:
            f.write(str(x)+'\n')
        f.write('begin data:\n')
        for l in L:
            if len(l)>0:
                f.write(list_str(l)+'\n')

def scrape(filepath):
	L = []
	with open(filepath,'r') as f:
		N = int(f.next())
		print N
		if N > 100:
			#these are too big
			return [],0
		#sometimes the first candidate is labeled '1', sometimes '0'
		offset = int(f.next()[0])
		for _ in range(N):
			f.next()
		for line in f:
			l = line.split(',')
			count = int(l[0])
			sig =[]
			i = 1
			while l[i][0]!='{' and i<len(l)-1:
				sig.append(int(l[i])-offset)
				i+=1
			for _ in range(count):
				#some election data had repeated "write-in" markers
				L.append(list(sig))
	return L,N


if __name__ == '__main__':
    np.set_printoptions(suppress=True, precision=3)
    path = os.getcwd()+os.sep+'unfiltered'
    save_path = os.getcwd()+os.sep+'filtered'
    for filename in os.listdir(path):
        filepath=path+os.sep+filename
        L,N = scrape(filepath)
        L,N = strongly_connected_restriction(L,N)
        write_soi(save_path+os.sep+filename,L,N)
