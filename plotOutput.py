import matplotlib.pyplot as plt
import numpy as np
from orderedBlock import importMyData
import copy



def plotDAGprobs(name):
    dataRaw = importMyData(name)
    for line in dataRaw:
        print line
    printdag = [[int(round(float(dataRaw[i][j])/10)) for j in range(len(dataRaw[0]))] for i in range(len(dataRaw[0]))]
    for line in printdag:
        print line
    print "######################"
    dataList = [[float(dataRaw[i][j])/10 for j in range(len(dataRaw[0]))] for i in range(len(dataRaw[0]))]
    for line in dataList:
        print line
    print "######################"
    data=np.array(dataList) 
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(data, cmap=plt.cm.Blues)
    ax.set_xticks(np.arange(data.shape[0])+0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[1])+0.5, minor=False)
    column_labels = [i for i in range(1,len(dataList)+1)]
    row_labels = [i for i in range(1,len(dataList)+1)]
    ax.set_xticklabels(row_labels, minor=False)
    ax.set_yticklabels(column_labels, minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    nameStrAr=name.split('.')
    plt.savefig(nameStrAr[0])

def importMyData(name):
    arr = []
    matrix=[]
    i = 0;
    for line in open(name):
        for value in line.rstrip('\n').split(','):
            arr.append(value)
        matrix.append(copy.deepcopy(arr))    
        arr=[]
    return matrix
plotDAGprobs("selectiveBest.csv")
plotDAGprobs("BOT.csv")

