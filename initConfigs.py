from Kernel import Kernel,Gibbs,Cyclic
from State import DagState,LatentVariableState

from DAGutil import generateData
from util import importMyData


def initData(n,arity,dagname):
    dag = importMyData(dagname)
    alpha = 0.5
    data = generateData(dag,n,alpha,arity)
    return data

def initLatentVariableState(data,model):    
    variableDomain="discrete"
    numbObjects=[] #unknown
    structuralConstraints = {}
    structuralConstraints["model"]=model
    structuralConstraints["numberVariables"]=12
    lvs = LatentVariableState(data, variableDomain,numbObjects,structuralConstraints)
    return lvs
    
def initDagState(data,lvs,arity):   
    variableDomain="discrete"
    numbObjects=[] #unknown
    structuralConstraints = {}
    structuralConstraints["z"]=lvs.z
    structuralConstraints["numberVariables"]=len(lvs.z)
    structuralConstraints["edgePriors"]=lvs.dens.edgePriors
    allRanges = [arity for i in range(len(data[0]))]
    structuralConstraints["ranges"]=allRanges
    ds = DagState(data, variableDomain,numbObjects,structuralConstraints)
    return ds

def initStates(data,arity,model):
    lvs =initLatentVariableState(data,model)
    ds=initDagState(data,lvs,arity)
    lvs.dens.setChildDensity(ds.dens)
    return lvs,ds



#gk = Gibbs(lvs, lvs.dens)