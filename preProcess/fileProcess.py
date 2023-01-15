import os
import pandas as pd
from preProcess.get_args import get_args

def readFile(path):  # 读取文件
    '''
    参数:
    - path: 文件路径[str]
    返回:
    - df: 文件内容
    '''
    fileType = path.split('.')[-1]
    if(fileType == "xlsx"):
        # df = pd.read_excel(path)
        df = pd.read_excel(path, header=None)
        df.rename(columns=df.iloc[0], inplace=True)
        df = df.dropna(how='all')  # 删除空白行
        # df = df.fillna(method='ffill')  # 将缺失值默认为上一行的值
    elif(fileType == "csv"):
        df = pd.read_csv(path)
    return df



def A():
    args = get_args()
    dataDir = args.data_dir  # data 目录
    fileLabel = args.fileLabel  # label 文件
    fileLabel = dataDir + fileLabel
    path = os.path.abspath(fileLabel)
    df = readFile(fileLabel)  # 读取标签文件{第j个切削刃的刀具磨损在第i个条件下标签，以及物件形式(离散值,one-hot转化为离散)}

    df.to_excel("test.xlsx", index=False)


    # print(os.walk(dataDir))
    for files in os.walk(dataDir):
        for name in files[2]:
            temp_names = files[0] + '/' + name
            all_names1.append(temp_names)
    pass