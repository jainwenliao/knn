import numpy
import matplotlib.pyplot as plt 
import kNN
from numpy import array
plt.rcParams['font.sans-serif'] = ['SimHei']

group, labels = kNN.createDataset()
datingDataMat,datingLabels = kNN.file_matrix('datingTestSet2.txt')

fig = plt.figure()
ax = fig.add_subplot(111)
#使用dataDataMat中的第二列，第三列数据
ax.scatter(datingDataMat[:,1], datingDataMat[:,2], 15.0*array(datingLabels), 15.0*array(datingLabels))
plt.xlabel("玩视频游戏所耗时间的百分比")
plt.ylabel("每周消费的冰淇淋公升数")

normMat, ranges, minvals = kNN.auto_Normalize(datingDataMat)


test_vectors = kNN.handwritingclasstest()
print(test_vectors)