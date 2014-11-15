from orderedBlock import *
import csv
from collections import Counter
#################################################################################################################
#################################################################################################################
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
def sampleDirichlet(alpha,arity):
    sample = [random.gammavariate(alpha,1) for i in range(arity)]
    sample = [v/sum(sample) for v in sample]
    return sample
def hist2class(hist):
    hz=[]
    for j in range(len(hist)):
        maxValue=0
        maxClass=0
        for i in range(len(hist)):
            if hist[i][j]>maxValue:
                maxValue=hist[i][j]
                maxClass=i
        hz.append(maxClass)
    return hz
def checkOver(number,over):
    if number>over:
        return 1
    else: 
        return 0
def writeMyCSV(name,matrix):
    with open(name, 'w') as fp:
        a = csv.writer(fp, delimiter=',')
        a.writerows(matrix)
        
def selectClass(zMatrix,OrderMatrix):
    out =[]
    for i in range(len(zMatrix)):
        for j in range(len(OrderMatrix[i])):
            for index,value in enumerate(zMatrix[i]):
                if value==OrderMatrix[i][j]:
                    zMatrix[i][index]=j+100
    for j in range(len(zMatrix[0])):
        data = Counter(column(zMatrix,j)) # Returns all unique items and their counts
        valueL = data.most_common(1)
        value,c=valueL[0]
        out.append(value-100)
    return out
def multMatrixScalar(matrix,scalar):
    return [[matrix[i][j]*scalar for j in range(len(matrix[0]))] for i in range(len(matrix))]
        
#################################################################################################################
#################################################################################################################
#################################################################################################################

dag = importMyData("groundTruth.csv")
print "Ground truth:"
for line in dag:
    print line
print "###############"
n = 200
alpha = 0.5
arity = 2
allRanges = [arity for i in range(len(dag))]
numIterations = 2000
data = generateData(dag,n,alpha,arity)
iter_modelAv=10
condition = "orderedBlock_n="+str(n)+"_moves="+str(numIterations)
print condition

numVar = 12
print numIterations


accZ=[[0 for i in  range(iter_modelAv)]for j in range(numVar)]

od=[[0 for i in range(numVar)]for j in range(numVar)]
gd =[[0 for i in range(numVar)]for j in range(numVar)]
orderList=[]
zList=[]
scores=[]
setAllDAGs_Best=[]
setAllDAGs_GIBBS=[]
bestOfTen=0
botIndex =0
for select_index in range(iter_modelAv):
    print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
    print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
    print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
    print select_index
    outdag,z,bd,currentHist,bo,bestZ,bestDAGscore,gibbsdag=spsl(numIterations,data,[arity for i in range(12)])
    if bestDAGscore>bestOfTen:
        bestOfTen=bestDAGscore
        botIndex=select_index
    orderList.append(bo)
    zList.append(bestZ)
    scores.append(math.exp(bestDAGscore))
    setAllDAGs_Best.append(bd)
    setAllDAGs_GIBBS.append(gibbsdag)
    print "######################"
    for line in outdag:
        print line
    print "######################"
    printdag = [[int(round(gibbsdag[i][j])) for j in range(len(gibbsdag[i]))] for i in range(len(gibbsdag[i]))]
    for line in printdag:
        print line
    print "######################"
    od=add(bd, od)
    gd=add(gibbsdag,gd)
    print "######################"
    print "bestDAG"
    for line in bd:
        print line
    print "%%%%%%%%%%%%%%%%%%%%%"
    print "Histogramm"
    print hist2class(currentHist)
    print "%%%%%%%%%%%%%%%%%%%%"
    print "bestZ"
    print bestZ
    print "%%%%%%%%%%%%%%%%%%%%"
    print "Best Order"
    print bo
print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
print "Gibbs"
printdag = [[int(round(gd[i][j])) for j in range(len(gd[i]))] for i in range(len(gd[i]))]
for line in printdag:
    print line
print "######################"
printdag = [[int(round(float(gd[i][j])/iter_modelAv)) for j in range(len(gd[i]))] for i in range(len(gd[i]))]
for line in printdag:
    print line
print "############################################################"
print "Best"
print "############################################################"
printdag = [[int(round(od[i][j])) for j in range(len(od[i]))] for i in range(len(od[i]))]
for line in printdag:
    print line
print "######################"
print "######################"

printdag = [[int(round(float(od[i][j])/iter_modelAv)) for j in range(len(od[i]))] for i in range(len(od[i]))]
for line in printdag:
    print line
print "######################"
print selectClass(zList,orderList)
scores = normVector(scores)
selectiveBest=[[0 for i in range(numVar)]for j in range(numVar)]
for i in range(len(setAllDAGs_Best)):
    selectiveBest=add(selectiveBest,multMatrixScalar(setAllDAGs_Best[i],scores[i]))
selectiveGibbs=[[0 for i in range(numVar)]for j in range(numVar)]
for i in range(len(setAllDAGs_GIBBS)):
    selectiveGibbs=add(selectiveGibbs,multMatrixScalar(setAllDAGs_GIBBS[i],scores[i]))
print "##############"
print "selectiveBest"
for line in selectiveBest:
    print line
print scores

writeMyCSV("selectiveBest.csv",selectiveBest)
writeMyCSV("BOT.csv",setAllDAGs_Best[botIndex])

print condition
