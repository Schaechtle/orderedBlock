'''
Created on 17 Nov 2014

@author: ulli
'''
import random
import copy
from State import DagState,LatentVariableState
from util import sampleDiscrete,normVector
from cycleChecker import cyclic
from DAGutil import getParents
class Kernel(object):
    '''
    BLAISE-style Kernel
    '''


    def __init__(self, state, rootDensity):
        '''
        Constructor
        '''
    ######################################################################      
    def sampleDiscrete(self,probs): # samples discrete values from a distribution
        sortedProbabilites = [i[1] for i in sorted(enumerate(probs), key=lambda x:x[1])]
        indeces = [i[0] for i in sorted(enumerate(probs), key=lambda x:x[1])]
        randUni=random.uniform(0,1)
        #print randUni
        for i in range(len(probs)):
            if i>0:
                sortedProbabilites[i]=sortedProbabilites[i]+sortedProbabilites[i-1]
            if randUni<sortedProbabilites[i]:
                break
        return indeces[i]
    ######################################################################             

class Cyclic(Kernel):
    '''
    Cyclic Moves between Kernels
    '''
    
    def __init__(self,kernels, states, densities):
        '''
        Constructor
        '''
        self.kernels=kernels
        self.sates=states
        self.densities=densities

    
    def SampleNextState(self,kernels, states,densities):
        newStates=[]
        for i in range(len(kernels)):
            newStates.append(kernels[i].SampleNextState(states[i],densities[i]))
            


        
class MH(Kernel):
    '''
    Metropolis-Hastings
    ''' 
    def __init__(self, state, rootDensity):
        '''
        Constructor
        '''
        Kernel.__init__(self,state, rootDensity)
    
    def SampleNextState(self):
        print("MH not implemented yet")
        return None

    
class Gibbs(Kernel):
    '''
    Gibbs Sampling
    '''    
    def __init__(self, state, rootDensity):
        '''
        Constructor
        '''
        Kernel.__init__(self,state, rootDensity)
        self.proposalKernel = ProposalKernel(state,rootDensity)
        self.moves=[]
        self.scores=[]
    
    def SampleNextState(self,state, rootDensity):
        moves = self.proposalKernel.enumPosMoves(state, rootDensity)
        density =[]
        for move in moves:
            density.append(rootDensity.moveTarget(move))  
        choose=sampleDiscrete(normVector(density))
        self.moves.append(moves[choose])     
        self.scores.append(density[choose])            
        return moves[choose]    
    
class ProposalKernel(Kernel):
    def __init__(self, state, rootDensity):
        '''
        Constructor
        '''
        Kernel.__init__(self,state, rootDensity)
    
    def enumPosMoves(self,state,rootDensity):
        if  isinstance(state,DagState):
                toggled = False   
                newDag = copy.deepcopy(state.dag)
                randomRange1 = range(len(state.dag))
                randomRange2=randomRange1
                random.shuffle(randomRange1)
                random.shuffle(randomRange2)
                while not toggled:
                    for i_index in randomRange1:
                        for j_index in randomRange2:
                            if rootDensity.edgePriors[i_index][j_index]>0:
                                if (state.dag[i_index][j_index]==1)&(random.uniform(0,1)>rootDensity.edgePriors[i_index][j_index]):
                                    newDag[i_index][j_index]=0
                                    toggled = True
                                    break
                                if state.dag[i_index][j_index]==0&(random.uniform(0,1)<rootDensity.edgePriors[i_index][j_index]):
                                    if len(getParents(state.dag, j_index))<4:
                                        newDag[i_index][j_index]=1 
                                        toggled = True
                                        break                
                        if toggled:
                            if state.checkCyclic==False:
                                break
                            else:
                                if (not cyclic(newDag)):
                                    break
                                else:
                                    newDag = copy.deepcopy(state.dag)
                                    toggled=False
                return [state.dag,newDag]
        else:
            if isinstance(state,LatentVariableState):
                if not state.numbObjects:
                    if state.structuralConstraints["model"]=="orderedBlock":
                        
                        #rootDensity.varIndex+=1
                        #rootDensity.crpSampler(state.alpha,state.m,state.dens.varIndex)
                        #moves=[i for i in range(2*len(rootDensity.z)+1)]
                        return rootDensity.enumPossibleMoves()
                    if state.structuralConstraints["model"]=="block": 
                        return rootDensity.enumPossibleMoves()
                
    
    def SampleNextState(self):
        print("Not implemented")
        return None
    