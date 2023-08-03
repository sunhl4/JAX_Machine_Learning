'''
Created on Sep 16, 2010
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)

Output:     the most popular class label

@author: pbharri
'''
import jax
import jax.numpy as jnp
import numpy as onp
import operator
from os import listdir
from numpy import *

def CreateDataSet():
    group = jnp.array([[1.0,1.1],[1.0,1.0],[0.0,0.0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels
def Classfy0(inX, DataSet, Labels, k):
    # DataSetSize : The length of first dimension(shape[0]) of training set (DataSet) or the number of training data
    DataSetSize = DataSet.shape[0]
    #tile(inX ,(DataSetSize,1)) :Copy inX to DataSetSize rows and 1 column. return a item which have the same size with training set (DataSet)
    Difference_Data = onp.tile(inX, (DataSetSize, 1)) - DataSet
    Square_Difference_Data = Difference_Data**2
    #Sum at the second dimension of quare_Difference_Data (eg:x^2+y^2+z^2)
    Square_Distance = Square_Difference_Data.sum(axis=1)
    Distance = Square_Distance**0.5
    #Sort Distance(Euclidean distance) at ascending order. return the index(Distance[i] = Labels[i]) of Distance
    SortedDistIndicies = jnp.argsort(Distance)
    ClassCount = {}
    for i in range(k):
    #     i=i-Nearest-Neighbors; VoteLabel[j] = j-Nearest-Neighbors
        VoteLabel = Labels[SortedDistIndicies[i]]
    #change to dic
        ClassCount[VoteLabel] = ClassCount.get(VoteLabel, 0) + 1
    #sort SortedClassCout(list) by second index. (becuase dic is disorder)
    SortedClasssCount = sorted(ClassCount.items(), key=operator.itemgetter(1), reverse=True)
    return SortedClasssCount[0][0]

def File2Matrix(filename):
    fr = open(filename)
    ArrayLines = fr.readlines()
    NumberLines = len(ArrayLines)
    ReturnData = zeros((NumberLines,3))
    ClassLabelVector = []
    index = 0
    for line in ArrayLines:
        line = line.strip()
        ListFromLine = line.split('\t')
        ReturnData[index,:] = ListFromLine[0:3]
#        ClassLabelVector.append(int(ListFromLine[-1]))
        ClassLabelVector.append(ListFromLine[-1])

        index +=1

    for i in range(NumberLines):
        if ('didntLike' == ClassLabelVector[i]):
            ClassLabelVector[i] = 1
        elif ('smallDoses' == ClassLabelVector[i]):
            ClassLabelVector[i] = 2
        else:
            ClassLabelVector[i] = 3
    return ReturnData, ClassLabelVector


def AutoNormalization(DataSet):
    MinValue = DataSet.min(0)
    MaxValue = DataSet.max(0)
    LargestDistance = MaxValue - MinValue
    NormalizationDataSet = zeros(shape(DataSet))
    #Number of Data
    n = DataSet.shape[0]
    NormalizationDataSet = DataSet - onp.tile(MinValue,(n,1))
    NormalizationDataSet = NormalizationDataSet / onp.tile(LargestDistance,(n,1))
    return NormalizationDataSet, LargestDistance, MinValue

def DatingClassTest():
    TestRatio = 0.10
    DatingDataSet, DatingLabels = File2Matrix('datingTestSet.txt')
    NormalizationDataSet, LargestDistance, MinValue =  AutoNormalization(DatingDataSet)
    n = NormalizationDataSet.shape[0]
    NumberTestData = int(n*TestRatio)
    ErrorCount = 0.0
    for i in range (NumberTestData):
        ClassifierResult = Classfy0(NormalizationDataSet[i,:], NormalizationDataSet[NumberTestData:n,:],\
                                    DatingLabels[NumberTestData:n],3)
        print ("the classifier came back with: %d, the real answer is: %d" %(ClassifierResult, DatingLabels[i]))
        if (ClassifierResult != DatingLabels[i]):
            ErrorCount += 1.0
    print ("the total error rate is: %f" % (ErrorCount/float(NumberTestData)))

def ClassifyPerson():
    LabelList = ['not at all', 'in small doses', 'in large doses']
    VideoGamesTime = float(input("percentage of time spent playing video games?"))
    FrequentFilerMiles = float(input("frequent filer miles earned per year?"))
    IceCream = float(input("liters of ice cream consumed per year?"))
    DatingDataSet, DatingLabels = File2Matrix('datingTestSet.txt')
    NormalizationData, LargestDistance, MinValue  = AutoNormalization(DatingDataSet)
    # print(type(MinValue))
    InArray = onp.array([FrequentFilerMiles, VideoGamesTime, IceCream])
    # print(type(InArray))
    ClassifierResult = Classfy0((InArray - MinValue)/LargestDistance, NormalizationData, DatingLabels, 3)
    print ("You will probably like this person: %s " % LabelList[ClassifierResult -1])











if __name__ == '__main__':
    #---------------------2.1.1
    # group ,labels = CreateDataSet()
    #---------------------2.1.2
    # KNN = Classfy0([0,0], group, labels ,3)
    # print(KNN)
    #---------------------2.2.1
    # Datas, Labels = File2Matrix('datingTestSet.txt')
    # print (Datas)
    # print (Labels)
    #---------------------2.2.3
    # Datas,Labels = File2Matrix('datingTestSet.txt')
    # NormalData, LargestDistance, MinValue = AutoNormalization(Datas)
    # print(NormalData)
    # print(LargestDistance)
    # print(MinValue)
    #---------------------2.2.4
    # DatingClassTest()
    # ---------------------2.2.5
    ClassifyPerson()



