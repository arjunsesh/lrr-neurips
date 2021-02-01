import numpy as np
import networkx as nx
import pickle

D = pickle.load(open('nascar-scraped.p','rb'))
L = D['train-lists']
L2 = D['test-lists']
n = D['N']
A = np.zeros((n,n))
for l in L:
    for idx in range(len(l)):
        x=l[idx]
        for y in l[idx:]:
            A[y,x]=1

for l in L2:
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
with open('nascargcc.txt','w') as f:
    f.write(str(gcc)[5:-2]+'\n')
    with open('nascar2002.txt','r') as g:
        f.write(g.next())
        for line in g:
            driver = int(line.split(' ')[0])
            if driver in gcc:
                f.write(line)
