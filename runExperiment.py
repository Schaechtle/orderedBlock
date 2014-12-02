from Kernel import Gibbs
from util import maxScoresIndeces,normVector,add,multMatrixScalar,writeMyCSV

# default
model="block"


import sys
   
     

global reStart
period =100
reStart=True
numIterations = 2000
n = 75
statesPerVar = 2
nBest=100
repetitions=10


for i in range(1,len(sys.argv)):
    if str(sys.argv[i])=="-p":
        model = str(sys.argv[i+1])
    if str(sys.argv[i])=="-s":
        numIterations = int(sys.argv[i+1])
    if str(sys.argv[i])=="-rs":
        # problems with converting from argument to bool
        if (sys.argv[i+1]=="True"):
            reStart = True
        else:
            reStart = False
    if str(sys.argv[i])=="-rsp":
        period = int(sys.argv[i+1])
if (model=="block") | (model=="orderedBlock"):
    from initConfigs import  initStates,initData
    blockStructure=True
else:
    from initConfigNotOrdered import  initStates,initData
    blockStructure=False
    if (model=="sparse"):
        prior=True
        model="sparse Prior"
    else:
        prior=False
        model="uniform Prior"


print("Model: "+model)
print("Number Moves: "+str(numIterations))
if reStart:
    print("Periodic Restart after: "+str(period))



collDags=[]
collScores=[]
data = initData(n,statesPerVar,"groundTruth.csv")

numVariables=len(data[0])


def runMCMC(numIterations,lvs,ds):
    for i in range(numIterations):
        if ((i%period)==0)& reStart:
            dagTemp=ds.dens.dags
            scoreTemp=ds.dens.scores
            lvs,ds=initStates(data,statesPerVar,model)
            ds.dens.scores= scoreTemp
            ds.dens.dags=dagTemp
        for j in range(1):
            move=gk.SampleNextState(lvs, lvs.dens)
            lvs.update(move)
        move=gkDAG.SampleNextState(ds, ds.dens)
        ds.update(move)
    return ds.dens.dags,ds.dens.scores


def runMCMConDAGonly(numIterations,ds):
    for i in range(numIterations):
        if (i%period)==0 & reStart:
            dagTemp=ds.dens.dags
            scoreTemp=ds.dens.scores
            ds=initStates(data,statesPerVar,prior)
            ds.dens.scores= scoreTemp
            ds.dens.dags=dagTemp
        move=gk.SampleNextState(ds, ds.dens)
        ds.update(move)
    return ds.dens.dags,ds.dens.scores


if blockStructure:
    for i in range(repetitions):
        print(i)
        lvs,ds=initStates(data,statesPerVar,model)
        gk = Gibbs(lvs, lvs.dens)
        gkDAG = Gibbs(ds, ds.dens)
        dags,scores=runMCMC(numIterations,lvs,ds)
        collDags.extend(gkDAG.moves)
        collScores.extend(gkDAG.scores)
else:        
    for i in range(repetitions):
        print("repetion: "+str(i))
        ds=initStates(data,statesPerVar,prior)
        gk = Gibbs(ds, ds.dens)
        dags,scores=runMCMConDAGonly(numIterations,ds)
        collDags.extend(gk.moves)
        collScores.extend(gk.scores)   


scores,indeces = maxScoresIndeces(collScores,nBest)  
scoresOld =scores 
scores = normVector(scores)
selectiveBest=[[0 for i in range(numVariables)]for j in range(numVariables)]

for i in range(nBest):
    selectiveBest=add(selectiveBest,multMatrixScalar(collDags[indeces[i]],scores[i]))    


print("##############")
print("selectiveBest")
for line in selectiveBest:
    print(line)
theBest=collDags[indeces[0]]
print("##############")
print("theBest")
for line in theBest:
    print(line)



writeMyCSV("selectiveBest.csv",selectiveBest)        
        
    



