import os
import random
import numpy as np
import pandas as pd
from scipy import interpolate
from torch.utils.data import Dataset, DataLoader
from preProcess.get_args import get_args
from preProcess.Mydataset import Mydataset

def readFile(path):  # 读取文件
    '''
    参数:
    - path: 文件路径[str]
    返回:
    - df: 文件内容
    '''
    fileType = path.split('.')[-1]
    if(fileType == "xlsx"):  # 读取标签文件
        df = pd.read_excel(path, header=None)
        df.rename(columns=df.iloc[0], inplace=True)
        df = df.dropna(how='all')  # 删除空白行
        df = df.fillna(method='ffill')  # 将缺失值默认为上一行的值
    elif(fileType == "csv"):
        df = pd.read_csv(path, header=None)
        df.columns = ['Axial Force','Bending Moment of X','Bending Moment of Y','Torsion of Z','Channel 1','Channel 2', 'Spindle power', 'Spindle current']
    return df

def readX(dataDir):  # 获取自变量数据集
    '''
    参数: 
    - 数据目录[str]
    返回:
    - dataX: [[序号,材质,df]]
    '''
    classname = next(os.walk(dataDir))[1]  # 工件类型
    dataX = []  # [[序号,材质,df]]
    for files in os.walk(dataDir):
        for name in files[2]:
            if files[0].split('\\')[-1] in classname:
                df = readFile(dataDir + '/' + files[0].split('\\')[-1] + '/' + name)
                # df.to_csv('test.csv', index=False)
                dataX.append([name.split('.')[0], files[0].split('\\')[-1], df])
    return dataX

def readY(dataDir):  # 读取因变量数据集
    '''
    参数: 
    - 数据文件[str]
    返回:
    - dataY: [[序号,材质,df]]
    '''
    dataY = []
    df = readFile(dataDir)  # 读取标签文件{第j个切削刃的刀具磨损在第i个条件下标签，以及物件形式(离散值,one-hot转化为离散)}
    for index, row in df.iterrows():
        if(row['Number'] != '1'):
            dataY.append([row['Number'], row['W1'], row[1:5]])
    return dataY

def interDf(df): # 插值
    '''
    参数:
    -  df
    返回:
    - dfY: 因变量集
    '''
    tempX = np.arange(len(df))  # X范围
    tempY1, tempY2, tempY3, tempY4 = df['Blade-1'], df['Blade-2'], df['Blade-3'], df['Blade-4']  # 4序列
    tempY1null, tempY2null, tempY3null,  tempY4null = tempY1.isnull(), tempY2.isnull(), tempY3.isnull(), tempY4.isnull()  # 是否空
    intempXeff = tempX[tempY1null]  # 无效数据x
    tempXeff = np.setdiff1d(tempX, intempXeff, assume_unique=False)  # 有效数据x
    tempYeff1, tempYeff2, tempYeff3, tempYeff4 = tempY1.values[tempXeff], tempY2.values[tempXeff], tempY3.values[tempXeff], tempY4.values[tempXeff]  # 有效数据y
    ktemp1 = [(tempYeff1[i] - tempYeff1[i-1])/(tempXeff[i] - tempXeff[i-1]) for i in range(1, len(tempXeff))]  # 增长率
    ktemp2 = [(tempYeff2[i] - tempYeff2[i-1])/(tempXeff[i] - tempXeff[i-1]) for i in range(1, len(tempXeff))]
    ktemp3 = [(tempYeff3[i] - tempYeff3[i-1])/(tempXeff[i] - tempXeff[i-1]) for i in range(1, len(tempXeff))]
    ktemp4 = [(tempYeff4[i] - tempYeff4[i-1])/(tempXeff[i] - tempXeff[i-1]) for i in range(1, len(tempXeff))]
    
    eff = -1  # 指向增量的索引
    temp_k1, temp_k2, temp_k3, temp_k4 = 0, 0, 0, 0  # 当前增量
    temp_i = -1 # 当前移动
    temp_y1, temp_y2, temp_y3, temp_y4 = 0, 0, 0, 0  # 当前数量值
    Y1, Y2, Y3, Y4 = [], [], [], []
    for i in range(len(tempX)):
        if i in tempXeff:  # x自变量有效
            print(i)
            Y1.append(tempY1.values[i])
            Y2.append(tempY2.values[i])
            Y3.append(tempY3.values[i])
            Y4.append(tempY4.values[i])

            eff += 1
            try:
                temp_k1, temp_k2, temp_k3, temp_k4 = ktemp1[eff], ktemp2[eff], ktemp3[eff], ktemp4[eff]  # 当前增量
            except:
                pass
            temp_i = 0  # 移动量置零
            temp_y1, temp_y2, temp_y3, temp_y4 = tempY1.values[i], tempY2.values[i], tempY3.values[i], tempY4.values[i]
        else: # 无效
            temp_i += 1
            Y1.append(temp_y1 + temp_k1 * temp_i)
            Y2.append(temp_y2 + temp_k2 * temp_i)
            Y3.append(temp_y3 + temp_k3 * temp_i)
            Y4.append(temp_y4 + temp_k4 * temp_i)
    # print(np.array([Y1,Y2,Y3,Y4]).T)
    dfX = df.loc[:,'Number': 'Spindle current']  # 获取因变量
    dfY = pd.DataFrame(np.array([Y1,Y2,Y3,Y4]).T, columns=['Blade-1','Blade-2','Blade-3','Blade-4'])  # 获取自变量
    # print(dfY)
    XlisT = dfX.values.T.tolist()
    YlisT = dfY.values.T.tolist()
    arrLisT = (np.array(XlisT + YlisT)).T
    df = pd.DataFrame(arrLisT)
    df.to_csv('df.csv', encoding='utf-8',index=True)
    # print(dfX)
    return df

def dfGenerator(dataX, dataY):  # 生成数据
    df = pd.DataFrame(columns=['Number', 'W1',
    'Axial Force','Bending Moment of X','Bending Moment of Y','Torsion of Z','Channel 1','Channel 2', 'Spindle power', 'Spindle current', 
    'Blade-1', 'Blade-2', 'Blade-3', 'Blade-4'])  # 生成空df
    idx = 0
    for xList in dataX:  # 遍历dataX
        dfTemp = pd.DataFrame(columns=['Number', 'W1','Axial Force','Bending Moment of X','Bending Moment of Y','Torsion of Z','Channel 1','Channel 2', 'Spindle power', 'Spindle current', 'Blade-1', 'Blade-2', 'Blade-3', 'Blade-4'])  # 生成空df
        xLoc = dfTemp.shape[0]  # 新的位置
        dfTemp = dfTemp.assign(**xList[2])  # 填入因变量
        dfTemp['Number'] = pd.Series([xList[0]]*len(xList[2]))
        dfTemp['W1'] = pd.Series([xList[1]] * len(xList[2]))
        dfTemp.loc[xLoc, ['Blade-1', 'Blade-2', 'Blade-3', 'Blade-4']] = [lst for lst in dataY if lst[1] == xList[1] and lst[0] == int(xList[0])][0][2].values
        
        # 测试用
        idx += 1
        if(idx % 10 == 0):
            print('idx = ' + str(idx))
        # 测试用

        df = pd.concat([df, dfTemp])
    
    # dfX = df.loc[:,'Number': 'Spindle current']
    # dfY = interDf(df)  # 插值
    # df.to_csv('df.csv', encoding='utf-8',index=True)
    # dfX.to_csv('dfX.csv', encoding='utf-8',index=True)
    # dfY.to_csv('dfY.csv', encoding='utf-8',index=True)
    # return dfX, dfY
    df = interDf(df)
    return df

def scramble_data(lisDf):  # 打乱数据
    '''
    参数:
    - text: [[int]]
    返回: 
    - text[0]: 
    '''
    cc = list(zip(lisDf))  # 创建一个包含文本中原始字符的图元列表；zip为制作元组
    random.seed(100)  # 随机种子
    random.shuffle(cc)  # 随机每个列表
    lisDf[:] = zip(*cc)  # 解压为列表
    return lisDf[0]

def setGenerator(df):  # 数据集生成器
    '''
    参数:
    - df: 全部数据
    返回:
    - train_x: 训练集自变量
    - train_y: 训练集因变量
    - valid_x: 验证集自变量
    - valid_y: 验证集因变量
    - test_x:  测试集自变量
    - test_y:  测试集因变量
    '''
    args = get_args()
    fold = args.fold
    batch_size = args.batch_size
    set_size = args.set_size

    lisDf = df.values.tolist()  # 转换为列表
    da = scramble_data(lisDf)  # 打乱数据
    print('rand the data')

    x=[]
    for j in range(set_size):  # 将 da 分成五等份切片，分为训练集、验证集、测试集
        x.append(da[(len(da)*j)//set_size: (len(da)*(j+1))//set_size])
    train=[]
    valid=x[fold%set_size]  # 从 x 中取出一个部分并赋值给验证集
    test=x[(fold+1)%set_size]  # 从 x 中取出另一个部分并赋值给测试集
    for i in range(2,5):  # 取出剩余部分分给训练集
        train+=x[(i+fold)%5]
    print('train valid test')

    train_x=np.array(train)[:,0:10]  # 自变量,16序列
    print('train_x')
    train_y=np.array(train)[:,10:]  # 因变量，APP编号
    print('train_y')
    valid_x=np.array(valid)[:,0:10]
    print('valid_x') 
    valid_y=np.array(valid)[:,10:]
    print('valid_y') 
    test_x=np.array(test)[:,0:10]
    print('test_x') 
    test_y=np.array(test)[:,10:]
    print('test_y')  
    return train_x, train_y, valid_x, valid_y, test_x, test_y

def oneHot(key):
    if   key == 'W1':
        return [1, 0, 0, 0, 0, 0, 0, 0, 0]
    elif key == 'W2':
        return [0, 1, 0, 0, 0, 0, 0, 0, 0]
    elif key == 'W3':
        return [0, 0, 1, 0, 0, 0, 0, 0, 0]
    elif key == 'W4':
        return [0, 0, 0, 1, 0, 0, 0, 0, 0]
    elif key == 'W5':
        return [0, 0, 0, 0, 1, 0, 0, 0, 0]
    elif key == 'W6':
        return [0, 0, 0, 0, 0, 1, 0, 0, 0]
    elif key == 'W7':
        return [0, 0, 0, 0, 0, 0, 1, 0, 0]
    elif key == 'W8':
        return [0, 0, 0, 0, 0, 0, 0, 1, 0]
    elif key == 'W9':
        return [0, 0, 0, 0, 0, 0, 0, 0, 1]
    else:
        return [0, 0, 0, 0, 0, 0, 0, 0, 0]

def setArrInt(arr):  # 将数组中的str全部转换为字符串
    '''
    参数:
    - arr: 二维numpy, stype = str
    返回:
    - arr: 二维numpy, stype = float
    '''
    arr = arr.T
    if len(arr) == 10:
        arrA = arr[0]
        arrB = np.array([oneHot(i) for i in arr[1]])
        arrC = arr[2:]
        arr = np.vstack((arrA.T, arrB.T, arrC))
    arr = arr.astype(np.float)
    arr = arr.T
    return arr

def dataLoad(df):  # 加载器生成
    '''
    参数:
    - df: 数据
    返回:
    - train_dataloader: 训练集加载器
    - valid_dataloader: 验证集加载器
    - test_dataloader: 测试集加载器
    '''
    args = get_args()
    batch_size = args.batch_size

    train_x, train_y, valid_x, valid_y, test_x, test_y = setGenerator(df)
    train_x, train_y, valid_x, valid_y, test_x, test_y = setArrInt(train_x), setArrInt(train_y), setArrInt(valid_x), setArrInt(valid_y), setArrInt(test_x), setArrInt(test_y)
    train_dataloader = []  # 训练数据加载器
    valid_dataloader = []  # 验证数据加载器
    test_dataloader = []  # 测试数据加载器
    # 这里要把x中的非数值换成数字，大概是.T翻转之后再用列表生成器换一下，不管了睡觉
    train_dataloader.append(
        DataLoader(Mydataset(np.array(train_x), np.array(train_y)),
                    batch_size=batch_size,
                    shuffle=True, num_workers=0))
    valid_dataloader.append(
        DataLoader(Mydataset(np.array(valid_x), np.array(valid_y)),
                    batch_size=batch_size,
                    shuffle=True, num_workers=0))
    test_dataloader.append(
        DataLoader(Mydataset(np.array(test_x), np.array(test_y)),
                    batch_size=batch_size,
                    shuffle=True, num_workers=0))
    
    # train_dataloader = np.squeeze(train_dataloader)  
    # for idx, (data, target) in enumerate(train_dataloader): 
    # for i in train_dataloader: 
    #     pass
    # for idx, (data, target) in enumerate(test_dataloader): 
    #     print(target[0])
    return train_dataloader,valid_dataloader,test_dataloader

def one_hot(df):
    print(df)  
    pass

def fileProcess():
    # args = get_args()
    # dataDir = args.data_dir  # data 目录
    # fileLabel = args.fileLabel  # label 文件
    # fileLabel = dataDir + fileLabel
    # dataX = readX(dataDir)  # [[材质, 序号 ,df]]
    # dataY = readY(fileLabel)  # [[材质, 序号 ,df]]
    # df = dfGenerator(dataX, dataY)  # 因变量，自变量
    print('begin')
    df = pd.read_csv('df.csv', header=0, index_col=0)  # 所有数据
    print('read the csv')
    # df = one_hot(df)
    train_dataloader,valid_dataloader,test_dataloader = dataLoad(df)
    return [train_dataloader,valid_dataloader,test_dataloader]