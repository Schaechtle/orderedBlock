from util import column,sampleDirichlet,sampleDiscrete
import itertools
import copy
 ######################################################################
def getParents(dag,i):
    col = column(dag, i)
    return [index for index in range(len(col)) if col[index]==1 ]   
######################################################################
def count(parentConfig,data,childIndex,parentIndeces,ranges):
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
def matchConfig(line,parents):
    return tuple([line[p] for p in parents])    
######################################################################
def initDag(v):
    return [[0 for j in range(v)] for i in range(v)]
######################################################################
def getParentConfig(ranges): #get configuration of parents for counts
    if not ranges:
        return []
    parentvalues =[]
    for maxValue in ranges:
        allValues= []
        for i in range(0,maxValue):
            allValues.append(i)
        parentvalues.append(allValues)
    return itertools.product(*parentvalues) 
#################################################################################################################

def generateData(dag,n,alpha,arity):
    data = [[0 for i in range(len(dag))]for j in range(n)]
    allRanges = [arity for i in range(len(dag))]
    for i in range(len(dag)):
        myDict = {}
        parents = getParents(dag, i)
        probs=sampleDirichlet(alpha,arity)
        if not parents:
            for j in range(n):
                data[j][i]=sampleDiscrete(probs)
        else:
            pc = getParentConfig([allRanges[index] for index in parents])
            for item in pc:
                    myDict[item]=sampleDirichlet(alpha,arity)
            for j in range(n):
                data[j][i]=sampleDiscrete(myDict[matchConfig(data[j], parents)])
    return data

###################################################################### 

