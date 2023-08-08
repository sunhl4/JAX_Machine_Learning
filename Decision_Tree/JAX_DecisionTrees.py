'''
Created on Oct 12, 2010
Decision Tree Source Code for Machine Learning in Action Ch. 3
@author: Peter Harrington
'''
from math import log
import operator
import jax.numpy as jnp

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    #change to discrete values
    return dataSet, labels

def CalcShannonENT(DataSet):
    NumberEntries = len(DataSet)
    LabelCounts = {}
    for FeatVec in DataSet:
        CurrentLabels = FeatVec[-1]
        if CurrentLabels not in LabelCounts.key[]
            LabelCounts[CurrentLabels] = 0
        LabelCounts[CurrentLabels] += 1
    ShannonEnt = 0.0
    for key in LabelCounts:
        probability = float(LabelCounts[key])/NumberEntries
        ShannonEnt -= prob * log(prob,2)
    return ShannonEnt

def SplitDataSet(DataSet, Axis, Value):
    ResetDataSet = []
    for FeatVec in DataSet:
        if FeatVec[Axis] == Value:
            ReduceFeatVec = FeatVec[:Axis] #FeatVec[0:0] 即取0～-1的数据，所以为空
            # print (ReduceFeatVec)
            ReduceFeatVec.extend(FeatVec[Axis+1:])
            ResetDataSet.append(ReduceFeatVec)
    return ResetDataSet

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1      #the last column is used for the labels
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):        #iterate over all the features
        featList = [example[i] for example in dataSet]#create a list of all the examples of this feature
        uniqueVals = set(featList)       #get a set of unique values
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy     #calculate the info gain; ie reduction in entropy
        if (infoGain > bestInfoGain):       #compare this to the best gain so far
            bestInfoGain = infoGain         #if better than current best, set to best
            bestFeature = i
    return bestFeature                      #returns an integer

def ChooseBestFeatureToSplit(DataSet):
    NumberFeatures = len(DataSet[0]) - 1
    BaseEntropy = CalcShannonENT(DataSet)
    BestInformationGain = 0.0, BestFeature = -1
    for i in range(NumberFeatures):
        FeatureList = [example[i] for example in DataSet]
        UniqueValuse = set(FeatureList)
        NewEntropy = 0.0
        for Value in UniqueValuse:
            SubDataSet = SplitDataSet(DataSet,i,Value)
            Probility = len(SubDataSet)/float(len(DataSet))
            NewEntropy += Probility * CalcShannonENT(SubDataSet)
        InformationGain = BaseEntropy - NewEntropy
        if (InformationGai:n > BestInformationGain):
            BestInformationGain = InformationGain
            BestFeature = i
    return BestFeature


def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]#stop splitting when all of the classes are equal
    if len(dataSet[0]) == 1: #stop splitting when there are no more features in dataSet
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]       #copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree)[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel

def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)

if __name__ == '__main__':
    DataSet, Labels = createDataSet()
    SplitDataSet(DataSet,0,1)