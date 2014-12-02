'''
Created on 19 Nov 2014

@author: ulli
'''
from Density import Density
from util import shiftList,mult,betaFunction
from DAGutil import initDag
import math
import copy
from priors import computeEdge,priorG,orderPrior
class BDensity(Density):
    '''
    classdocs
    '''
    def __init__(self,alpha, beta, z,  m,  kplus,classOrder,  numberVariables):
        '''
        Constructor
        '''
        self.alpha=alpha
        self.beta=beta
        self.z=z
        self.m=m
        self.kplus = kplus 
        self.numberVariables=numberVariables
        self.classOrder=classOrder
        self.cePrior = self.classEdgePrior(self.beta,self.kplus,self.classOrder)
        self.edgePriors=self.transformEdgeDensity(self.cePrior, self.z)
        self.currentIndex=0
        
    def moveTarget(self,move):
        return self.moveLookUp[move]
    
    def enumPossibleMoves(self):
        self.moveLookUp={}
        self.currentIndex=self.getCurrentIndex() #hack, to be removed    
        self.previousZ=self.z[self.currentIndex]
        pz = self.crpSampler(self.alpha,self.m,self.currentIndex)
        mi = copy.deepcopy(self.m)
        zi = copy.deepcopy(self.z)
        lambd=[]
        ikplus=self.kplus
        if  len(pz)<=len(self.z):
            numberClasses=len(pz)
        else:
            numberClasses=len(pz-1)
        for i in range(numberClasses):
            zi[self.currentIndex]=i
            mi[self.previousZ]-=1
            if self.m[self.previousZ]==0:
                mi.pop(self.previousZ)
                mi.append(0)
                zi = shiftList(self.previousZ,self.zi)
                if i<len(pz)-1:
                    ikplus-=1
            else:
                ikplus+=1              
            nplus,nminus = computeEdge(self.childDensity.dag,zi,ikplus)
            pGz = priorG(self.beta,nplus,nminus)
            lambd.append(pGz)
        probVector =mult(pz, lambd)
        out=[]
        for i in range(len(probVector)):
            out.append(i)
            self.moveLookUp[i]=probVector[i]
        return out
        #####################################################     
            #####################################################     
    def updateState(self,newClass):       # I leave this in here to be called from the state
        if newClass>self.kplus-1:
            self.z[self.currentIndex]=self.kplus
            self.kplus+=1
        else:
            self.z[self.currentIndex]=newClass               
        self.m[int(self.z[self.currentIndex])]+=1
        self.m[self.previousZ]-=1            
        if self.m[self.previousZ]==0:
            self.m.pop(self.previousZ)
            self.m.append(0)
            self.z = shiftList(self.previousZ,self.z)
            self.kplus-=1
        nplus,nminus = computeEdge(self.childDensity.dag,self.z,self.kplus)    
        self.edgePriors=self.getEdgeDensity(nplus,nminus,self.z)       
            ######################################################################  
    
    
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
                    no[i][j]=( math.pow(priorOrder,(self.beta-1))*math.pow((1-priorOrder),self.beta))/betaFunction(self.beta,self.beta) 
        return no
    ######################################################################
    def getCurrentIndex(self):
        self.currentIndex+=1
        if self.currentIndex>=self.numberVariables-1:
            self.currentIndex=0
        return self.currentIndex
    
        ######################################################################
    def getInitialDensity(self,cePrior,z):
        edgeDensity=initDag(len(z))
        for i_index in range(len(z)):
            for j_index in range(len(z)):
                    edgeDensity[i_index][j_index]=cePrior[z[i_index]][z[j_index]]
        return edgeDensity
    ######################################################################
    def getEdgeDensity(self,nplus,nminus,z):
        edgeDensity=initDag(len(z))
        for i_index in range(len(z)):
            for j_index in range(len(z)):
                    edgeDensity[i_index][j_index]=math.log(betaFunction(self.beta + nplus[z[i_index]][z[j_index]],self.beta + nminus[z[i_index]][z[j_index]])/betaFunction(self.beta,self.beta))
        return edgeDensity
    ######################################################################
    def transformEdgeDensity(self,cePrior,z):
        edgeDensity=initDag(len(z))
        for i_index in range(len(z)):
            for j_index in range(len(z)):
                    edgeDensity[i_index][j_index]=cePrior[z[i_index]][z[j_index]]
        return edgeDensity
