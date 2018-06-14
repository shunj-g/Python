from numpy import *
'''
m1 = dataSet[nonzero(dataSet[:,3] > [['4'],
                                     ['4'],
                                     ['4'],])[0],:][0]
'''
data = [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
dataSet = [['1','2','3','4'],
           ['5','6','7','8'],
           ['9','10','11','12']]
#shape(dataSet)获取行值，并且的得到行和列的值
K = mat(zeros((5,1)))
print(K[0].copy() )
print(K)
'''

i = 0
list = [example[i][0] for example in dataSet]
for j in list:
    print(j)
'''
def splitDataSet(dataSet,axis,value):
    retDataSet = []
    for featVec in dataSet:
        print(featVec)
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet
retDataSet = splitDataSet(data ,2,'yes')
print(retDataSet)