## DEPRECATED
import torch
import numpy as np
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

class BP(nn.Module):
    """
    Implementation of the Bastell-Polking k-th order model as a pytorch module
    """
    def __init__(self,n,k,d):
        """
        Initializes a k-th order Batsell-Polking model

        Args:
        n- number of items in universe
        k- order of the model
        d- rank of the model
        """
        super(BP,self).__init__()
        shape = tuple([n]*k)
        self.U = nn.Parameter(torch.nn.init.normal_(torch.Tensor(*shape)))
        #self.F = nn.Parameter(torch.nn.init.normal(torch.Tensor(n,k,d)))
        self.k = k
        self.n = n
        self.d= d
        self.m = nn.LogSoftmax()

    def forward(self,x):
        """
        Computes choice probabilities for k-th order BP model
        """
        utils = self.U
        for _ in range(self.k-1):
            utils = torch.matmul(utils,torch.squeeze(x))
        utils = x*utils+(1-x)*(-16)
        return self.m(utils)

    def __str__(self):
        return 'BP:k='+str(self.k)

class CDM(nn.Module):
    """
    Implementation of the CDM choice model as a Pytorch module
    """
    def __init__(self,n,d):
        """
        Initializes a CDM model

        Args:
        n- number of items in the universe
        d- number of dimensions for feature and context embeddings
        """
        super(CDM, self).__init__()
        self.fs = nn.Parameter(torch.nn.init.normal(torch.Tensor(n,d)))
        self.cs = nn.Parameter(torch.nn.init.normal(torch.Tensor(d,n)))
        self.m = nn.LogSoftmax()
        self.d = d
        self.n = n

    def forward(self,x):
        """
        computes the CDM choice probabilities P(.,S)

        Args:
        x- indicator vector for choice set S, i.e. a 'size(S)-hot' encoding of the
        choice set, or a batch of these
        """
        #compute the utilities of each alternative for the given choice set
        u = x*torch.sum(torch.mm(self.fs,x*self.cs),dim=1)+(1-x)*(-16)

        #softmax of utils gives probs
        p = self.m(u)
        return p

    def __str__(self):
        return 'CDM-d='+str(self.d)#+',ue='+str(self.ue)

class PCMC(nn.Module):
    """
    Implementation of the PCMC choice model as a pytorch module
    """
    def __init__(self,n,batch=True,ep=10**(-3)):#,gamma=None):
        """
        Initializes a PCMC model

        Args:
        n- number of alternatives in universe
        batch- whether we are doing batch SGD, which plays strangely with PCMC
        ep- minimum allowed entry of Q, 0 entries can lead to a disconnected Markov chain
        #gamma- MNL parameters that can be used to initialize a PCMC model,
                currently deprecated
        """
        super(PCMC, self).__init__()

        #TODO: add functionality to initialize with MNL params gamma, template:
        #if gamma is None:
            #Q = torch.nn.init.uniform(torch.Tensor(n,n),.4,.6)
        #else:
            #print 'code is deprecated, using random Q'
            #Q = torch.nn.init.uniform(torch.Tensor(n,n),.4,.6)

        #randomly initialize Q
        Q = torch.nn.init.uniform(torch.Tensor(n,n),.4,.6)
        self.Q = nn.Parameter(Q)
        self.epsilon = ep
        self.batch = batch
        self.n = n

        #cache vectors used for solving stationary distribution
        self.b = torch.zeros(n)
        self.b[-1]=1
        self.I = torch.eye(n)
        self.m = nn.LogSoftmax()

    def forward(self,x):
        """
        computes the PCMC choice probabilities P(.,S)

        Args:
        x- indicator vector for choice set S, i.e. a 'size(S)-hot' encoding of the
        choice set, or a batch of such encodings
        """
        #batching vectorizes in a peculiar way for PCMC, but works automatically
        #for other models, so store whether we plan to use batch at model initialization
        #and encode it here
        if self.batch:

            L = [] #empty list that will later contain choice probs

            for S in x.split(1):
                S=x
                S_mat = torch.mm(torch.t(S),S)
                Q = torch.clamp(self.Q*S_mat,min=self.epsilon)
                for i in range(self.n-1):
                    Q[i,i]=-torch.sum(Q[i,:])+Q[i,i]
                Q[:,self.n-1]=S
                b = Variable(torch.zeros(self.n)+self.epsilon)
                b[-1]=1
                pi,LU = torch.gesv(b,torch.t(Q))
                pi = S*torch.t(pi)+1e-16
                L.append(torch.log(pi/torch.sum(pi)))
            return torch.cat(L)
        else:
            S=x
            S_mat = torch.mm(torch.t(S),S)
            Q = torch.clamp(self.Q*S_mat,min=self.epsilon)
            for i in range(self.n-1):
                Q[i,i]=-torch.sum(Q[i,:])+Q[i,i]
            Q[:,self.n-1]=S
            b = Variable(torch.zeros(self.n,1))
            b[-1]=1
            pi,LU = torch.solve(b,torch.t(Q))
            pi = S*torch.t(pi)+1e-16

            return torch.log(pi/torch.sum(pi))

    def __str__(self):
        return 'PCMC'

class MNL(nn.Module):
    """
    Implementation of MNL choice model as a Pytorch module
    """
    def __init__(self,n):
        super(MNL, self).__init__()
        self.u = nn.Parameter(torch.nn.init.normal(torch.Tensor(n)))
        self.n = n
        self.m = nn.Softmax()

    def forward(self, x):
        """
        computes the PCMC choice probabilities P(.,S)

        Args:
        x- indicator vector for choice set S, i.e. a 'size(S)-hot' encoding of the
        choice set, or a batch of such encodings
        """
        u = x*self.u+(1-x)*-16
        p = self.m(u)
        return torch.log(p/torch.sum(p))

    def __str__(self):
        return 'MNL'

if __name__ == '__main__':
    # some testing code
    np.set_printoptions(suppress=True,precision=3)
    n = 5;ns=200;d=2;mnl_data=False
    X = np.zeros((ns,n))
    Y = np.empty(ns).astype(int)
    gamma = np.random.rand(n)
    gamma/= np.sum(gamma)
    #print gamma
    for i in range(ns):
        S=np.random.choice(range(n),size=np.random.randint(2,n),replace=False)
        for j in S:
            X[i,j]=1
        if mnl_data:
            P = gamma[S]
            P/= np.sum(P)
        else:
            P = np.random.dirichlet(np.ones(len(S)))
        c = np.random.choice(S,p=P)
        Y[i]=int(c)

    X = torch.Tensor(X)
    Y = torch.LongTensor(Y)
    dataset = torch.utils.data.TensorDataset(X,Y)
    dataloader = torch.utils.data.DataLoader(dataset)#,batch_size=2)
    criterion = nn.NLLLoss()
    #models = [MNL(n),PCMC(n),CDM(n,d),BP(n=n,k=3,d=2)]


    models = [BP(n=n,k=3,d=2)]
    U_0 = models[0].parameters().next()

    for model in models:
        if str(model)=='PCMC':
            utils = models[0].parameters().next().data.numpy()
            g= np.exp(utils)
            g/= np.sum(g)
            #model = PCMC(n,gamma=g,batch=False)
        optimizer = optim.Adam(model.parameters())
        #optimizer = optim.SGD(model.parameters(), lr=0.001)#, momentum=0.9)

        for epoch in range(20):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(dataloader, 0):
                # get the inputs
                inputs, labels = data
                # wrap them in Variable
                inputs, labels = Variable(inputs), Variable(labels)
                #print 'choice set'
                #print inputs

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                #print 'NLL losses'
                #print outputs
                #print 'utils'
                #print model.parameters().next().data
                #assert np.random.rand()<.99
                loss = criterion(outputs, labels)

                #if np.isnan(loss.data[0]):
                #    print inputs
                #    print outputs
                #    print labels
                #    assert False
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.data[0]
                if i % 200 == 199:    # print every 200 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 200))
                    running_loss = 0.0

            print(str(model)+' Finished Training')

    for model in models:
        for x in model.parameters():
            print(x)
