'''
Created on 10 Nov 2014

@author: ulli
'''
import csv
import copy
import random

import math

def importMyData(name):
    matrix=[]
    i = 0;
    for line in open(name):
        arr=[]
        for value in line.rstrip('\n').split(','):
            arr.append(float(value))
        matrix.append(arr)    
    return matrix

######################################################################
#                            Numerical/Algebraic Stuff               #
######################################################################   
def column(matrix,i): #returns a column of a matrix
    return [row[i] for row in matrix]
######################################################################
def matrixGammaLog(matrix):
    logmatrix=copy.deepcopy(matrix)
    if isinstance(matrix,list):
        if isinstance(matrix[0],list):
            for i in range(len(matrix)):
                for j in range(len(matrix[0])):
                    logmatrix[i][j]=math.lgamma(matrix[i][j])
        else:
            for i in range(len(matrix)):
                logmatrix[i]=math.lgamma(matrix[i])
    else:
        logmatrix = math.lgamma(matrix)
    return logmatrix
######################################################################
def add(first_in,second_in):
    out=copy.deepcopy(first_in)
    if isinstance(first_in,list):           
        if isinstance(first_in[0], list):
            for i in range(len(first_in)):
                for j in range(len(first_in[0])):
                    out[i][j]=first_in[i][j]+second_in[i][j]
        else:
            for i in range(len(first_in)):
                out[i]=first_in[i]+second_in[i]
        return out
    else:
        return first_in+second_in
######################################################################    
def subtract(first_in,second_in):
    out=copy.deepcopy(first_in)
    if isinstance(first_in,list):           
        if isinstance(first_in[0], list):
            for i in range(len(first_in)):
                for j in range(len(first_in[0])):
                    out[i][j]=first_in[i][j]-second_in[i][j]
        else:
            for i in range(len(first_in)):
                out[i]=first_in[i]-second_in[i]
        return out
    else:
        return first_in-second_in   
######################################################################
    
def sum2(matrix_in):
    matrix=copy.deepcopy(matrix_in)
    if isinstance(matrix_in, list):
        if isinstance(matrix[0],list):
            return [sum(item) for item in matrix]
        else:        
            return sum(matrix)        
    return matrix
######################################################################

def sum22(matrix_in):
    matrix=copy.deepcopy(matrix_in)
    if isinstance(matrix_in, list):
        if isinstance(matrix[0],list):
            return [sum(column(matrix, i)) for i in range(len(matrix[0]))]
        else:        
            return sum(matrix)
def mult(fact1,fact2):return [x * y for x, y in zip(fact1, fact2)]
######################################################################
def normVector(myVector): 
    total = sum(myVector)
    if total==0:
        return [0.0 for x in myVector]  
    return [x/total for x in myVector]
######################################################################
def shiftList(fromValue,myList):
    for i in range(len(myList)):
        if myList[i]>fromValue:
            myList[i]-=1
    return myList
######################################################################
def order(kplus):
    s=list(range(kplus))
    random.shuffle(s) 
    return s
######################################################################
def betaFunction(a,b):
    return math.gamma(a)*math.gamma(b)/math.gamma(a+b)
######################################################################
def hist4z(currentHist,z):
    newHist=copy.deepcopy(currentHist)
    for i in range(len(z)):
        newHist[z[i]][i]+=1
    return newHist
def importMyDataInt(name):
    arr = []
    matrix=[]
    i = 0;
    for line in open(name):
        for value in line.rstrip('\n').split(','):
            arr.append(int(value))
        matrix.append(copy.deepcopy(arr))    
        arr=[]
    return matrix
######################################################################      
def sampleDiscrete(probs): # samples discrete values from a distribution
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
def sampleDirichlet(alpha,arity):
    sample = [random.gammavariate(alpha,1) for i in range(arity)]
    sample = [v/sum(sample) for v in sample]
    return sample

def maxScoresIndeces(L,n):
    newL = sorted(range(len(L)), reverse=True, key=lambda i:L[i])
    indeces=[newL[i] for i in range(n)]
    outL=[L[indeces[i]] for i in range(n)]
    return outL,indeces
def multMatrixScalar(matrix,scalar):
    return [[matrix[i][j]*scalar for j in range(len(matrix[0]))] for i in range(len(matrix))]
def writeMyCSV(name,matrix):
    with open(name, 'w') as fp:
        a = csv.writer(fp, delimiter=',')
        a.writerows(matrix)
def multList(L):
    out=1
    for item in L:
        out*=item
    return out
def binomialCoefficient(n, k):
    if k > n / 2:
        k = n - k
    result = 1.0
    for i in range(1,k+1):
        result *= ((n - (k - i)) / float(i))
    return int(result)

