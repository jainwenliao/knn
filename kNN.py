import os,sys
from numpy import *
import operator


def createDataset():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])#数据集
    labels = ['A','A','B','B']#标签
    return group, labels


def classify0(inX, dataSet, labels, k):
    #输入向量inX, 训练样本集dataset
    dataSet_size = dataSet.shape[0] #尺寸的维数4
    diffMat = tile(inX,(dataSet_size,1))-dataSet#求差值，tile将inX重复dataset_size次
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)#axis=1，每行相加。axis=0每列相加
    distances = sqDistances**0.5
    sorted_index = distances.argsort()#将元素从小到大排列,并提取出其下标

    classcount= {}
    for i in range(k):
        i_label = labels[sorted_index[i]] #i对于的标签
        #存入当前label以及对于的类别值，然后票数加1.如果不存在返回0
        classcount[i_label] = classcount.get(i_label,0)+1

    #把分类结果进行排序，然后返回票数多的
    sorted_classescount= sorted(classcount.items(),key = operator.itemgetter(1), reverse=True)

    return sorted_classescount[0][0]

def file_matrix(filename):
    fr = open(filename)#打开文件
    array_lines = fr.readlines() #读取文件所有行，保存在一个列表变量中
    numbers_lines = len(array_lines) #行数
    return_matrix = zeros((numbers_lines,3))#numbers_line行，3列的零矩阵

    classlabelvertor = []
    index = 0
    for line in array_lines:
        line = line.strip() #去掉所有的回车符
        list_line = line.split('\t') #用制表符把字符串隔开
        return_matrix[index,:] = list_line[0:3]#得到前三列数据
        classlabelvertor.append(int(list_line[-1]))#将最后一列元素加到向量classlabelvertor里
        index += 1

    return return_matrix, classlabelvertor

def auto_Normalize(dataSet):#归一化函数
    #将每列的最小值放到min_values中
    min_values = dataSet.min(0)
    #将每列的最大值放到max_values中
    max_values = dataSet.max(0)
    #求差范围
    ranges = max_values - min_values
    #得到一个和dataSet一样行，列，值都是0的数组
    normalize_dataset = zeros(shape(dataSet))
    m = dataSet.shape[0] #这个数组的行数
    
    normalize_dataset = dataSet - tile(min_values,(m,1))
    normalize_dataset = normalize_dataset / tile(ranges,(m,1))

    return normalize_dataset, ranges,min_values

#错误次数统计
def datingClassTest():
    #预留出0.1用于测试
    hoRatio = 0.1
    #读取数据和标签
    datingDataMat, datinglabels =file_matrix('datingTestSet2.txt')
    #数据归一化，范围，最小值
    normMat, ranges, minValue = auto_Normalize(datingDataMat)
    #行数
    m = normMat.shape[0]
    #测试数据的长度
    numTestVecs = int(m*hoRatio)
    errorcount = 0.0

    for i in range(numTestVecs):
        #normMat[i,:] 取出第i行的所有数据
        #normMat[numTestVecs:m,:]取出numTestVecs之后到m的每行数据
        #datinglabels[numTestVecs:m]取出numtestvecs之后到m的每行标签
        classify_result = classify0(normMat[i,:],normMat[numTestVecs:m,:], datinglabels[numTestVecs:m],3)

        #print("the classify_result come back with: " + classify_result + ", "+ 'the real answer is '+datinglabels[i]+' .')
        print("the classify_result came back with: %d, the real answer is:%d" % (classify_result,datinglabels[i]))

        if classify_result != datinglabels[i]:
            errorcount += 1

    error_rate = errorcount / float(numTestVecs)

    print("The total error rate is %f" % (error_rate))

def classify_person():
    result_list = ['not at all', 'in sall doses', 'in large doses',]#感兴趣的程度
    #询问
    percent_tats  = float(input("percentage of time spent playing video games?"))
    ff_miles = float(input("frequent flier playing video games?"))
    ice_cream = float(input("liters of ice cream consumed per year?"))
    #读取数据和标签
    datingDataMat, datinglabels = file_matrix("datingTestSet2.txt")
    #归一化
    normMat, ranges, min_values = auto_Normalize(datingDataMat)
    #将输入值转为数组
    in_arr = array([ff_miles, percent_tats,ice_cream])
    
    classify_result = classify0((in_arr-min_values)/ranges,normMat,datinglabels,3)

    print("You will pronably like this person: ", result_list[classify_result - 1])


#将32*32的二进制图像矩阵转为1*1024的向量
def img_vector(filename):
    #创建一个1*1024的numpy数组，值全为0
    return_vector = zeros((1,1024))
    fr = open(filename)

    for i in range(32):
        #读取前32行数
        line_str = fr.readline()
        #将每行的前32个字符值存储在numpy数组中
        for j in range(32):
            return_vector[0,32*i+j] = int(line_str[j])

    return return_vector

#手写数字测试
def handwritingclasstest():
    #初始化手写数字标签
    hwlabels = []
    #获取训练目录
    traingFileList = os.listdir('trainingDigits')
    #训练目录的长度
    m = len(traingFileList)
    #创建一个m行，1024列的零矩阵
    trainingMat =zeros((m,1024))
    #开始提取训练集
    for i in range(m):
        #从文件名解析出分类数字
        filenamestr = traingFileList[i]
        #以.为分隔符，显示第一个
        filestr = filenamestr.split('.')[0]
        #得到0
        classnamstr = int(filestr.split('_')[0])
        #将其加到hw_labels里
        hwlabels.append(classnamstr)
        #加载图像
        trainingMat[i,:] = img_vector("E:/machine_learning/trainingDigits/%s" % filenamestr)

    testFileList = os.listdir('testDigits')
    errorcount = 0.0
    m_test = len(testFileList)
    for i in range(m_test):
        #测试数据
        filenamestr = testFileList[i]
        filestr = filenamestr.split('.')[0]
        classnumstr = int(filestr.split('_')[0])
        vector_test = img_vector('E:/machine_learning/testDigits/%s'% filenamestr)
        classify_result = classify0(vector_test , trainingMat, hwlabels,3)

        print('the classifier came back with: %d, the real answer is: %d' % (classify_result,classnumstr))

        if(classify_result != classnumstr):errorcount += 1 
    print("\nthe total number of errors is: %d" % errorcount)
    print('\nthe total error rate is: %f'% (errorcount/float(m_test)))