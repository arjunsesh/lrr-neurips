import numpy as np


with open('nascar.soi','w') as g:
    with open('nascargcc.txt','r') as f:
        gcc = map(int,f.next().split(','))
        race = 1
        s='1,'
        L = []
        g.write(str(len(gcc))+'\n')
        for i in range(len(gcc)):
            g.write(str(i)+'\n')
        g.write('data starts: \n') #there is a line for number of rankings here
        f.next()
        for line in f:
            L = line.split(' ')
            driver = L[0]
            r = int(L[1])
            if r!=race:
                race = r
                g.write(s[:-1]+'\n')
                s='1,'

            elif int(driver) in gcc:
                s+=str(gcc.index(int(driver)))+','
