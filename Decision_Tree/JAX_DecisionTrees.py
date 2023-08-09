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
        if CurrentLabels not in LabelCounts.keys():
            LabelCounts[CurrentLabels] = 0
        LabelCounts[CurrentLabels] += 1
    ShannonEnt = 0.0
    for key in LabelCounts:
        probability = float(LabelCounts[key])/NumberEntries
        ShannonEnt -= probability * log(probability,2)
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


def ChooseBestFeatureToSplit(DataSet):
    NumberFeatures = len(DataSet[0]) - 1
    BaseEntropy = CalcShannonENT(DataSet)
    BestInformationGain = 0.0
    BestFeature = -1
    for i in range(NumberFeatures):
        FeatureList = [example[i] for example in DataSet]
        UniqueValuse = set(FeatureList)
        # print(UniqueValuse)
        NewEntropy = 0.0
        for Value in UniqueValuse:
            SubDataSet = SplitDataSet(DataSet,i,Value)
            Probility = len(SubDataSet)/float(len(DataSet))
            NewEntropy += Probility * CalcShannonENT(SubDataSet)
        InformationGain = BaseEntropy - NewEntropy
        if (InformationGain > BestInformationGain):
            BestInformationGain = InformationGain
            BestFeature = i
    return BestFeature

def MajorityCnt(ClassList):
    ClassCount = {}
    for Vote in ClassCount:
        if Vote not in ClassList.keys():
            ClassCount[Vote] = 0
        ClassCount[Vote] += 1
    SotedClassCount = sorted(ClassCount.items(), key=operator.itemgetter(1), reverse=True)
    return SotedClassCount
def CreatTree(DataSet, Labels):
    #将dataSet中的数据先按行依次放入example中，然后取得example中的example[-1]元素，放入列表classList中
    #按行遍历，按列取数据的方法
    ClassList = [example[-1] for example in DataSet]
    if ClassList.count(ClassList[0]) == len(ClassList):
        return ClassList[0]
    if len(DataSet[0]) == 1:
        return MajorityCnt(ClassList)
    BestFeature = ChooseBestFeatureToSplit(DataSet)
    BestFeatureLabels = Labels[BestFeature]
    MyTree = {BestFeatureLabels:{}}
    del(Labels[BestFeature])
    FeatureValues = [example[BestFeature] for example in DataSet]
    UniqueValues = set(FeatureValues)
    for Value in UniqueValues:
        SubLabels = Labels[:]
        MyTree[BestFeature][Value] = ChooseBestFeatureToSplit(SplitDataSet(DataSet, BestFeature, Value), SubLabels)
    return MyTree

def classify(InputTree, FeatLabels, TestVec):
    FistStr = list(InputTree)[0]
    SecondDict = InputTree[FistStr]
    FeatIndex = FeatLabels.index(FistStr)
    key = TestVec[FeatIndex]
    ValueOfFeat = SecondDict[key]
    if isinstance(ValueOfFeat,dict):
        ClassLabel = Classify(ValueOfFeat,FeatLabels, TestVec)
    else:
        ClassLabel = ValueOfFeat
    return ClassLabel

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
    #--------
    DataSet, Labels = createDataSet()
    print(len(DataSet[0]))
    # SplitDataSet(DataSet,0,1)
    # print(len(DataSet))
    # print(CalcShannonENT(DataSet))
    # print(ChooseBestFeatureToSplit(DataSet))

