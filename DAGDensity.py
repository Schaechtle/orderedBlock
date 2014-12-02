'''
Created on 17 Nov 2014

@author: ulli
'''
import copy
from Density import Density
from util import sum2,sum22,add,subtract,matrixGammaLog,column,betaFunction,sampleDiscrete,binomialCoefficient
from DAGutil import getParents,count,getParentConfig,computeEdge,initDag
from cycleChecker import cyclic
import math
from priors import sparsityPrior
class DAGDensity(Density):
    '''
    classdocs
    '''
    global lookUp
    lookUp={}
    global gamma
    gamma =0.5
    def __init__(self,data,ranges,z=False,prior=False):       
        '''
        Constructor
        '''
        self.z=z
        self.prior=prior
        self.data=data
        self.ranges =ranges
        if isinstance(prior,list) & isinstance(z,list):
            self.dag = self.initialGraph(len(ranges),prior,z)
            self.edgePriors=prior
        else:
            if (prior) & (z==False): # general sparsity prior
                self.dag = self.initialSparseGraph(len(ranges))

                self.edgePriors=[[2.0/(len(ranges)-1) for i in range(len(ranges))]for j in range(len(ranges))]
                for i in range(len(ranges)):
                    self.edgePriors[i][i]=0
            else:
                self.dag = self.initialSparseGraph(len(ranges))
                self.edgePriors=[[0.33 for i in range(len(ranges))]for j in range(len(ranges))]
                for i in range(len(ranges)):
                    self.edgePriors[i][i]=0
            
        self.beta =0.5
        self.scores=[]
        self.dags=[]

    #          Scoring Graph + Parent Structure    
    def moveTarget(self,move):
        if self.z==False:
            if self.prior==False:
                currentScore = math.exp(self.scoreG(move,self.data,self.ranges)+(len(self.data)*len(self.data[0])/4))
                self.scores.append(currentScore)
                self.dags.append(move) 
            else:
                currentScore = math.exp(self.scoreG(move,self.data,self.ranges)+(len(self.data)*len(self.data[0])/4))*sparsityPrior(move)
                self.scores.append(currentScore)
                self.dags.append(move) 
        else:
            if self.prior:
                nplus,nminus = computeEdge(move,self.z,len(set(self.z)))  
                prior = self.priorG(self.beta,nplus,nminus)    
                currentScore =  math.exp(self.scoreG(move,self.data,self.ranges)+(len(self.data)*len(self.data[0])/4))* prior
                self.scores.append(currentScore)
                self.dags.append(move)
                '''
            else:
                currentScore = self.scoreG(move,self.data,self.ranges)
                self.scores.append(currentScore)
                self.dags.append(move) 
                '''
        return currentScore   
    
    def scoreG(self,linkMatrix,data,allRanges):
        llG = 0
        for j in range(len(linkMatrix)):
            parents = getParents(linkMatrix, j)
            s=str(j)+"_"+str(parents)
            if not lookUp.get(s, False):               
                pc = getParentConfig([allRanges[index] for index in parents])
                n_ijk =count(pc,data,j,parents,allRanges)
                myScore=self.score(n_ijk, self.dirichletPrior(n_ijk,gamma))    
                lookUp[s]=myScore
            else:
                myScore=lookUp[s]
            llG +=myScore 
        return llG
    ######################################################################
    def score(self,n_ijk,alpha_ijk_in):
        alpha_ijk = copy.deepcopy(alpha_ijk_in)
        prod_k = sum22(subtract(matrixGammaLog(add(n_ijk,alpha_ijk)),matrixGammaLog(alpha_ijk)))
        alpha_ij = sum22(alpha_ijk_in)
        n_ij = sum22(n_ijk)
        prod_ij = subtract(matrixGammaLog(alpha_ij), matrixGammaLog(add(alpha_ij,n_ij)))
        return sum2(add(prod_ij,prod_k))
    ######################################################################
    
    ######################################################################
    def count(self,parentConfig,data,childIndex,parentIndeces,ranges):
        if not parentIndeces:
            counts=[]
            for i in range(0,ranges[childIndex]):  
                counts.append(0);
            for j in range(0,len(data)):
                counts[data[j][childIndex]]+=1
            return counts
        hashmapList=[]
        hashmap = {}
        for confTuple in parentConfig:                    
            hashmap[confTuple]=0
        for i in range(0,ranges[childIndex]):            
                hashmapList.append((hashmap.copy()))
        for j in range(0,len(data)):
            parentValue= [data[j][k] for k in parentIndeces]
            hashmapList[data[j][childIndex]].update({tuple(parentValue):hashmapList[data[j][childIndex]].get(tuple(parentValue))+1})
        counts=[]
        for hm in hashmapList:   
            counts.append(hm.values())
        return counts     
    ######################################################################    
    def dirichletPrior(self,counts,alpha):
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
    ######################################################################   
    def priorG(self,beta,nplus,nminus):
        p =0
        k=len(nplus)
        for a in range(k):
            for b in range(k):
                p+=math.log(betaFunction(beta + nplus[a][b],beta + nminus[a][b])/betaFunction(beta,beta))
        return math.exp(p) 
######################################################################      
    def initialGraph(self,N,cep,z): #samples the initial graph from random order, graph structure    
        dag = initDag(N)
        for i in range(N):
            for j in range(N):
                dag[i][j]=sampleDiscrete([1-cep[i][j],cep[i][j]])
        return dag  
######################################################################
    def initialSparseGraph(self,numVar): 
        isCyclic=True
        dag=initDag(numVar)
        while(isCyclic):
            p = 2.0/(numVar-1) # as in Jones_etal_2005_Experiments in Stochastic Computation
            for i in range(numVar):
                for j in range(numVar):
                    if (i!=j)&(dag[j][i]==0):
                        dag[i][j]=sampleDiscrete([1-p,p])
            isCyclic=cyclic(dag)
        return dag
######################################################################
    def initialRandomDag(self,numVar): 
        isCyclic=True
        dag=initDag(numVar)
        while(isCyclic):
            p = 0.1 # as in Jones_etal_2005_Experiments in Stochastic Computation
            for j in range(numVar):
                for i in range(numVar):
                    if len(getParents(dag, i))<5:
                        if (i!=j) & (dag[j][i]!=1):
                            dag[j][i]=sampleDiscrete([1-p,p])
            print(cyclic(dag))
            isCyclic=cyclic(dag)
        return dag                
    def lambdaMessage(self,z,beta):
        kplus= len(set(z))
        nplus,nminus = computeEdge(self.dag,z,kplus)
        pGz = self.priorG(beta,nplus,nminus)
        return pGz

    def sparsityPrior(self,dag): # as in Jones_etal_2005_Experiments in Stochastic Computation
        eAbs=sum([sum(line) for line in dag])
        numVar=len(dag)
        beta = 2.0/(numVar-1)
        prior = math.pow(beta,eAbs)*math.pow((1-beta),binomialCoefficient(numVar,2)-eAbs)
        return prior                           