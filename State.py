'''
Created on 17 Nov 2014

@author: ulli
'''
#import abc
from DAGDensity import DAGDensity
from Density import Density
from OBDensity import OBDensity
from BDensity import BDensity
from util import order
from priors import classPrior
class State(object):
    '''
    Generic State, depending on initilasation
    '''


    def __init__(self,data, variableDomain,numbObjects,structuralConstraints):
        
        
        '''
        Constructor
        '''
        
        self.data=data
        self.variableDomain=variableDomain
        self.numbObjects=numbObjects
        self.structuralConstraints=structuralConstraints



class DagState(State):
    dens =  Density()
    def __init__(self,data, variableDomain,numbObjects,structuralConstraints):
        State.__init__(self,data, variableDomain, numbObjects, structuralConstraints)
        
        if variableDomain=="discrete":
            cptGamma = 0.5             
            self.dens = DAGDensity(data,structuralConstraints["ranges"],structuralConstraints["z"],structuralConstraints["edgePriors"])
            self.dag=self.dens.dag
        if structuralConstraints["z"]==False:
            self.checkCyclic=True
        else:
            self.checkCyclic=False
    def update(self,move):
        self.dag=move
        self.dens.dag=move

class LatentVariableState(State):
    dens =  Density()
    z = []
    m = []
    kplus =0
    classOrder =[]
    def __init__(self,data, variableDomain,numbObjects,structuralConstraints):
        State.__init__(self,data,  variableDomain, numbObjects, structuralConstraints)
        if variableDomain=="discrete":
            if not numbObjects:
                if structuralConstraints["model"]=="orderedBlock":
                    self.alpha =0.5
                    self.beta =1
                    self.structuralConstraints=structuralConstraints
                    self.numberVariables=structuralConstraints["numberVariables"]
                    self.initState(self.numberVariables)
                    self.dens=OBDensity(self.alpha,self.beta,self.z, self.m, self.kplus, self.classOrder,self.numberVariables)
                else:
                    if structuralConstraints["model"]=="block":
                        self.alpha =0.5
                        self.beta =1
                        self.structuralConstraints=structuralConstraints
                        self.numberVariables=structuralConstraints["numberVariables"]
                        self.initState(self.numberVariables)
                        self.dens=BDensity(self.alpha,self.beta,self.z, self.m, self.kplus,self.classOrder, self.numberVariables)

                        
                        
                    
                   
                    #self.dens.setVariables(self.z, self.m, self.kplus, self.classOrder)
                    #self.dens.cePriors= self.dens.classEdgePrior(self.beta, self.kplus, self.classOrder)
                    
    def initState(self,numberVariables):
        self.z= classPrior(self.alpha,numberVariables)
        self.m= [self.z.count(i) for i in range(numberVariables)]  
        self.kplus= len(set(self.z))
        self.classOrder=order(self.kplus) # initialise random order
        
                     
    def update(self,move):
        self.dens.updateState(move)
        self.z= self.dens.z
        self.m= self.dens.m
        self.kplus= self.dens.kplus
        self.classOrder=self.dens.classOrder
        self.cePrior = self.dens.cePrior
                                 
######################################################################   
                        
        
        
            
        
        