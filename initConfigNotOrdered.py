from State import DagState
from DAGutil import generateData
from util import importMyData


def initData(n,arity,dagname):
    dag = importMyData(dagname)
    alpha = 0.5
    data = generateData(dag,n,alpha,arity)
    return data


def initDagState(data,arity,prior):   
    variableDomain="discrete"
    numbObjects=[] #unknown
    structuralConstraints = {}
    structuralConstraints["z"]=False
    structuralConstraints["numberVariables"]=len(data[0])
    structuralConstraints["edgePriors"]=prior
    allRanges = [arity for i in range(len(data[0]))]
    structuralConstraints["ranges"]=allRanges

    ds = DagState(data, variableDomain,numbObjects,structuralConstraints)
    return ds
def initStates(data,arity,prior):
    ds=initDagState(data,arity,prior)
    return ds

