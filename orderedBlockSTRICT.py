import itertools
import math
import copy
import random 


def spsl(numIterations,data,ranges):	
    n = len(data)    # initialise
    bestDAGscore=-9e+50
    global lookUp
    lookUp={}   
    global alpha
    alpha = 0.5
    beta = 1
    period=100
    numberVariables=len(ranges) 
    z= classPrior(alpha,numberVariables)
    m= [z.count(i) for i in range(numberVariables)]  
    kplus= len(set(z))
    classOrder=order(kplus) # initialise random order
    dag = initialGraph(numberVariables, classEdgePrior(beta, kplus, classOrder), z)

    nunSamples=numIterations/2 #number gibbs
    gibbsZ=copy.deepcopy(z)   
    currentHist=initDag(numberVariables)
    gibbsDag = initDag(numberVariables)  

    for round in range(numIterations):
	if (round % period)==0:
	    z= classPrior(alpha,numberVariables)
            m= [z.count(i) for i in range(numberVariables)]  
            kplus= len(set(z))
            classOrder=order(kplus) # initialise random order
            dag = initialGraph(numberVariables, classEdgePrior(beta, kplus, classOrder), z)
        for i in range(numberVariables):            
            previousZ=z[i]
            pz = crpSampler(alpha,m,i)
            zi = copy.deepcopy(z)
            lambd=[]
            
            for j in range(len(pz)-1):
                zi[i]=j
                if kplus==numberVariables: 
                    numberClasses=len(pz)
                else:
                    numberClasses=len(pz)-1 
                nplus,nminus = computeEdge(dag,zi,numberClasses)
                pGz = priorG(beta,nplus,nminus)
                ceprior = classEdgePrior(beta,kplus,classOrder)
                ep = bernoulli(dag,ceprior,zi)
                lambd.append(pGz*ep)
            if kplus<numberVariables:                
                for k in range(len(classOrder)+1):
                    newClassOrder=copy.deepcopy(classOrder)
                    zi[i]=kplus        
                    newClassOrder.insert(k,kplus)                     
                    nplus,nminus = computeEdge(dag,zi,len(pz))
                    pGz2 = priorG(beta,nplus,nminus)        
                    ceprior2 = classEdgePrior(beta,kplus+1,newClassOrder)  
                    ep2 = bernoulli(dag,ceprior2,zi)                  
                    lambd.append(pGz2*ep2)
                    if k>0:
                        pz.append(pz[j+1])
            chanceNewClass=normVector(mult(pz, lambd))            
            newClass=sampleDiscrete(chanceNewClass) 
            if newClass>kplus-1:
                classOrder.insert(newClass-kplus,kplus)
                z[i]=kplus
                kplus+=1
            else:
                z[i]=newClass               
            m[z[i]]+=1
            m[previousZ]-=1            
            if m[previousZ]==0:
                m.pop(previousZ)
                classOrder.remove(previousZ)
                m.append(0)
                z = shiftList(previousZ,z)
                classOrder = shiftList(previousZ,classOrder)
                kplus-=1              
        k=len(set(z))    
        #score old dag with new classes
        nplus,nminus = computeEdge(dag,z,k)  
        prior = priorG(beta,nplus,nminus)    
        currentScore = (scoreG(dag,data,ranges)) + math.log(prior)   
        # sample edges   
        cepriorNew = classEdgePrior(beta,kplus,classOrder)  
        newDag=sampleEdge(dag,cepriorNew,z)
        #score new dag with new classes
        nplus2,nminus2 = computeEdge(newDag,z,k)  
        prior2 = priorG(beta,nplus2,nminus2)           
        newScore = (scoreG(newDag,data,ranges)) +math.log(prior2)    
        # MH like step
        mh = (math.exp(newScore/n))/(math.exp(currentScore/n))
        if mh>1:
            dag = newDag
        else:
            if random.uniform(0,1)<mh:
                dag = newDag
        if newScore>bestDAGscore:
            bestDag=dag
            bestOrder=classOrder
            bestZ=z
            bestDAGscore=newScore    
        if round>=(numIterations-nunSamples): # check if we're past burn-in    
            gibbsDag = add(gibbsDag, dag)
            gibbsZ = add(gibbsZ,z)
            currentHist=hist4z(currentHist,z)
                    
    return dag,z,bestDag,currentHist,bestOrder,bestZ,(math.exp(bestDAGscore/n)),[[float(gibbsDag[i][j])/nunSamples for j in range(numberVariables)] for i in range(numberVariables)]
######################################################################
#                        Sampler                                     #
######################################################################
def crpSampler(alpha,m,i): #chinese restaurant process sampler
    p=[]
    for k in range(len(m)):
        if m[k]>0:
            p.append(m[k]/(i+alpha))
        else:
            p.append(alpha/(i+alpha))
            break
    return p
######################################################################
def sampleEdge(dag,ceprior,z): # per-edge gibbs sampling. Toggling one at a time
    toggled = False   
    newDag = copy.deepcopy(dag)
    randomRange1 = range(len(dag))
    randomRange2=randomRange1
    random.shuffle(randomRange1)
    random.shuffle(randomRange2)
    while not toggled:
        for i_index in randomRange1:
            for j_index in randomRange2:
                if ceprior[z[i_index]][z[j_index]]>0:
                    if (dag[i_index][j_index]==1)&(random.uniform(0,1)>ceprior[z[i_index]][z[j_index]]):
                        newDag[i_index][j_index]=0
                        toggled = True
                        break
                    if dag[i_index][j_index]==0&(random.uniform(0,1)<ceprior[z[i_index]][z[j_index]]):
                        newDag[i_index][j_index]=1 
                        toggled = True
                        break                
            if toggled:
                break
    return newDag
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
def initialGraph(N,cep,z): #samples the initial graph from random order, graph structure    
    dag = initDag(N)
    for i in range(N):
        for j in range(N):
            dag[i][j]=sampleDiscrete([1-cep[z[i]][z[j]],cep[z[i]][z[j]]])
    return dag  
######################################################################
#          Scoring Graph + Parent Structure                          #
######################################################################
def score(n_ijk,alpha_ijk_in):
    alpha_ijk = copy.deepcopy(alpha_ijk_in)
    prod_k = sum22(subtract(matrixGammaLog(add(n_ijk,alpha_ijk)),matrixGammaLog(alpha_ijk)))
    alpha_ij = sum22(alpha_ijk_in)
    n_ij = sum22(n_ijk)
    prod_ij = subtract(matrixGammaLog(alpha_ij), matrixGammaLog(add(alpha_ij,n_ij)))
    return sum2(add(prod_ij,prod_k))
######################################################################
def scoreG(linkMatrix,data,allRanges):
    llG = 0
    for j in range(len(linkMatrix)):
        parents = getParents(linkMatrix, j)
        s=str(j)+"_"+str(parents)
        if not lookUp.get(s, False):               
            pc = getParentConfig([allRanges[index] for index in parents])
            n_ijk =count(pc,data,j,parents,allRanges)
            myScore=score(n_ijk, dirichletPrior(n_ijk,alpha))    
            lookUp[s]=myScore
        else:
            myScore=lookUp[s]
        llG +=myScore 
    return llG
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
def matchConfig(line,parents):
    return tuple([line[p] for p in parents])    
######################################################################
def initDag(v):
    return [[0 for j in range(v)] for i in range(v)]
######################################################################
#                            Priors                                  #
######################################################################

def getParents(dag,i):
    col = column(dag, i)
    return [index for index in range(len(col)) if col[index]==1 ]    ######################################################################
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
def priorG(beta,nplus,nminus):
    p =0
    k=len(nplus)
    for a in range(k):
        for b in range(k):
            p+=math.log(betaFunction(beta + nplus[a][b],beta + nminus[a][b])/betaFunction(beta,beta))
    return math.exp(p) 
######################################################################   
def classPrior(alpha,N):
    m = [0 for i in range(N)]
    gammaOfAlpha = math.gamma(alpha)
    m[0]=1.0
    z=[]
    z.append(0)
    kplus=1
    for i in range(1,N):
        kProduct=1.0
        for k in range(kplus):
            kProduct*=math.factorial(m[k]-1)
            newTableProb = math.pow(alpha, kplus)*gammaOfAlpha/(N+alpha)*kProduct
        newTable=sampleDiscrete([1-newTableProb,newTableProb,])
        kplus+=newTable
        m[kplus-1]+=1    
        z.append(kplus-1)
    return z
######################################################################    
def dirichletPrior(counts,alpha):
    if isinstance( counts[0],list):
        prior=[[0 for i in counts[0]] for j in counts]
        for j in range(len(counts[0])):
            denom = sum(column(counts, j))+alpha*len(counts)
            for i in range(len(counts)):
                prior[i][j]=(float(counts[i][j])+alpha)/denom
    else:
        total = sum(counts)+alpha*len(counts)
        prior = [(float(a+alpha))/total for a in counts]
    return prior
###################################################################### 
def bernoulli(dag,cep,z):
    prior = 1
    for i in range(len(dag)):
        for j in range(len(dag)):
            prior*= math.pow(cep[z[i]][z[j]], dag[i][j])   * math.pow(1-cep[z[i]][z[j]],1-dag[i][j])
    return prior
######################################################################
def classEdgePrior(beta,kplus,order):
    priorOrder = orderPrior(kplus)
    no = [[0 for i in range(kplus)] for j in range(kplus)]
    for i in range(kplus):
        for j in range(kplus):
            if order.index(j)==(order.index(i)+1):
                no[i][j]=( math.pow(priorOrder,(beta-1))*math.pow((1-priorOrder),beta))/betaFunction(beta,beta) 
    return no
######################################################################
def orderPrior(kplus):
    return 1.0/math.factorial(kplus)
######################################################################
#                            Numerical/Algebraic Stuff               #
######################################################################   
def column(matrix,i): #returns a column of a matrix
    return [row[i] for row in matrix]
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
def importMyData(name):
    arr = []
    matrix=[]
    i = 0;
    for line in open(name):
        for value in line.rstrip('\n').split(','):
            arr.append(int(value))
        matrix.append(copy.deepcopy(arr))    
        arr=[]
    return matrix
