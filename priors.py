import copy
import math
from util import column,betaFunction,sampleDiscrete,binomialCoefficient

def computeEdge(dag,z,k):
    # returns count for edges, usage example. How many edges from second class to first are present: nplus[1][0]
    nplus  = [[0 for i in range(k)] for j in range(k)]
    nminus=copy.deepcopy(nplus)
    for i in range(len(dag)):
        for j in range(len(dag[0])):
            if dag[i][j]==1:
                nplus[z[i]][z[j]]+=1
            else:
                nminus[z[i]][z[j]]+=1            
    return nplus,nminus
######################################################################   
def priorG(beta,nplus,nminus):
    p =0
    k=len(nplus)
    for a in range(k):
        for b in range(k):
            p+=math.log(betaFunction(beta + nplus[a][b],beta + nminus[a][b])/betaFunction(beta,beta))
    return math.exp(p) 
######################################################################   

def classPrior(alpha,N):
    m = [0 for i in range(N)]
    gammaOfAlpha = math.gamma(alpha)
    m[0]=1.0
    z=[]
    z.append(0)
    kplus=1
    for i in range(1,N):
        kProduct=1.0
        for k in range(kplus):
            kProduct*=math.factorial(m[k]-1)
        newTableProb = math.pow(alpha, kplus)*gammaOfAlpha/(alpha+N)*kProduct
        newTable=sampleDiscrete([1-newTableProb,newTableProb])
        kplus+=newTable
        m[kplus-1]+=1    
        z.append(kplus-1)
    return z
######################################################################    
def dirichletPrior(counts,alpha):
    if isinstance( counts[0],list):
        prior=[[0 for i in counts[0]] for j in counts]
        for j in range(len(counts[0])):
            denom = sum(column(counts, j))+alpha*len(counts)
            for i in range(len(counts)):
                prior[i][j]=(float(counts[i][j])+alpha)/denom
    else:
        total = sum(counts)+alpha*len(counts)
        prior = [(float(a+alpha))/total for a in counts]
    return prior
###################################################################### 
def bernoulli(dag,cep,z):
    prior = 1
    for i in range(len(dag)):
        for j in range(len(dag)):
            prior*= math.pow(cep[z[i]][z[j]], dag[i][j])   * math.pow(1-cep[z[i]][z[j]],1-dag[i][j])
    return prior

def orderPrior(kplus):
    return 1.0/math.factorial(kplus)
######################################################################
def sparsityPrior(dag): # as in Jones_etal_2005_Experiments in Stochastic Computation
    eAbs=sum([sum(line) for line in dag])
    numVar=len(dag)
    beta = 2.0/(numVar-1)
    prior = math.pow(beta,eAbs)*math.pow((1-beta),binomialCoefficient(numVar,2)-eAbs)
    return prior  
    
    