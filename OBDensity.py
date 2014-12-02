'''
Created on 18 Nov 2014

@author: ulli
'''
from Density import Density
from util import shiftList,mult,betaFunction
from DAGutil import initDag
import math
import copy
from priors import computeEdge,priorG,orderPrior,bernoulli
class OBDensity(Density):
    '''
    classdocs
    '''


    def __init__(self,alpha,beta,z,m,kplus,classOrder,numberVariables):
        '''
        Constructor
        '''
        self.alpha=alpha
        self.beta =beta
        self.z= z
        self.m= m
        self.kplus= kplus
        self.classOrder=classOrder
        self.numberVariables=numberVariables
        self.cePrior = self.classEdgePrior(self.beta,self.kplus,self.classOrder)
        self.edgePriors=self.getEdgeDensity(self.cePrior, z)
        self.moveLookUp={}
        self.currentIndex=0
    
    def moveTarget(self,move):
        return self.moveLookUp[move]
    def getCurrentIndex(self):
        self.currentIndex+=1
        if self.currentIndex>=self.numberVariables-1:
            self.currentIndex=0
        return self.currentIndex
        
    
    def enumPossibleMoves(self):
        self.moveLookUp={}
        self.currentIndex=self.getCurrentIndex() #hack, to be removed    
        self.previousZ=self.z[self.currentIndex]
        pz = self.crpSampler(self.alpha,self.m,self.currentIndex)
        zi = copy.deepcopy(self.z)
        lambd=[]
        
        for j in range(len(pz)-1):
            zi[self.currentIndex]=j
            if self.kplus==self.numberVariables: 
                numberClasses=len(pz)
            else:
                numberClasses=len(pz)-1 
            nplus,nminus = computeEdge(self.childDensity.dag,zi,numberClasses)
            pGz = priorG(self.beta,nplus,nminus)
            cePrior = self.classEdgePrior(self.beta,self.kplus,self.classOrder)
            ep = bernoulli(self.childDensity.dag,cePrior,zi)
            lambd.append(pGz*ep)
        if self.kplus<self.numberVariables:                
            for k in range(len(self.classOrder)+1):
                newClassOrder=copy.deepcopy(self.classOrder)
                zi[self.currentIndex]=self.kplus        
                newClassOrder.insert(k,self.kplus)                     
                nplus,nminus = computeEdge(self.childDensity.dag,zi,len(pz))
                pGz2 = priorG(self.beta,nplus,nminus)        
                ceprior2 = self.classEdgePrior(self.beta,self.kplus+1,newClassOrder)  
                ep2 = bernoulli(self.childDensity.dag,ceprior2,zi)                  
                lambd.append(pGz2*ep2)
                if k>0:
                    pz.append(pz[j+1])
        probVector =mult(pz, lambd)
        out=[]
        for i in range(len(probVector)):
            out.append(i)
            self.moveLookUp[i]=probVector[i]
        return out
        #####################################################     
    def updateState(self,newClass):       # I leave this in here to be called from the state
        if newClass>self.kplus-1:
            self.classOrder.insert(newClass-self.kplus,self.kplus)
            self.z[self.currentIndex]=self.kplus
            self.kplus+=1
        else:
            self.z[self.currentIndex]=newClass               
        self.m[int(self.z[self.currentIndex])]+=1
        self.m[self.previousZ]-=1            
        if self.m[self.previousZ]==0:
            self.m.pop(self.previousZ)
            self.classOrder.remove(self.previousZ)
            self.m.append(0)
            self.z = shiftList(self.previousZ,self.z)
            self.classOrder = shiftList(self.previousZ,self.classOrder)
            self.kplus-=1
        self.cePrior = self.classEdgePrior(self.beta,self.kplus,self.classOrder) ##added  
        self.edgePriors=self.getEdgeDensity(self.cePrior, self.z)  
        self.childDensity.edgePriors=self.edgePriors      
            ######################################################################     
    def setChildDensity(self,density): 
        self.childDensity=density       
        ######################################################################
    def crpSampler(self,alpha,m,i): #chinese restaurant process sampler
        p=[]
        for k in range(len(m)):
            if m[k]>0:
                p.append(m[k]/(i+alpha))
            else:
                p.append(alpha/(i+alpha))
                break
        return p
    ######################################################################
    ######################################################################
    def classEdgePrior(self,beta,kplus,order):
        priorOrder = orderPrior(kplus)
        no = [[0 for i in range(kplus)] for j in range(kplus)]
        for i in range(kplus):
            for j in range(kplus):
                if order.index(j)==(order.index(i)+1):
                    no[i][j]=( math.pow(priorOrder,(beta-1))*math.pow((1-priorOrder),beta))/betaFunction(beta,beta) 
        return no
    ######################################################################
    def getEdgeDensity(self,cePrior,z):
        edgeDensity=initDag(len(z))
        for i_index in range(len(z)):
            for j_index in range(len(z)):
                    edgeDensity[i_index][j_index]=cePrior[z[i_index]][z[j_index]]
        return edgeDensity
