#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from numpy import *
from matplotlib import pyplot as plt

from os import listdir

import operator
from fileinput import filename

#创建矩阵
def createDataSet():
    group = array([[10.0,11.1],[11.0,7.0],[10,8.0],[20,0.1]])
    labels = ['A','A','B','B']
    return group,labels


#k近邻算法 分类器
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

#从文件中读取数据转化为矩阵
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector



#图标形式展示分类结果
# fig = plt.figure()
#      
# ax = fig.add_subplot(111)
#  
# cValue = ['r','y','g','b','r','y','g','b','r']  
# 
# ax.scatter(group[:,0],group[:,1],c=cValue)
# 
# plt.show()

#矩阵的归一化方法
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))
    return normDataSet,ranges,minVals

#利用样本测试算法准确率
def classTest():
    hoRatio = 0.10
    dataMat,labels = file2matrix("TestSet.txt")
    normMat,ranges,minVals = autoNorm(dataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],
                                     labels[numTestVecs:m],3)
        print("the classifier came back with: %d, the real answer is %d" % (classifierResult,labels[i]))
        if(classifierResult != labels[i]) : errorCount += 1.0
    print("the total error rate is %f" % (errorCount/float(numTestVecs)))


 #最终执行 根据输入值分类
def classifyPerson():
    resultList = ['not at all','in small doses','in large doses']
    percentTats = float(input())
    ffMiles = float(input())
    icecream = float(input)
    dataMat,labels = file2matrix("TestSet.txt")
    normMat,ranges,minVals = autoNorm(dataMat)
    inArr = array([ffMiles,percentTats,icecream])
    classifierResult = classify0((inArr-minVals)/ranges, normMat, labels, 3)
    print("you will probably like this person: ",resultList[classifierResult-1])
    
    
    
#手写数字识别
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handWritingClassTest():
    hwLabels = []
    trainingFileList = listdir('E:/pythonprojects/knn/trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('E:/pythonprojects/knn/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult,classNumStr))
        if(classifierResult != classNumStr) : errorCount += 1.0
    print("\nthe total num of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount/float(mTest)))

handWritingClassTest()