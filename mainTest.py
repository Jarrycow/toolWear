import numpy as np
import torch
from mainTrain import setDevice

from preProcess.fileProcess import fileProcess

def TPFN(target, pre):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    target = target.tolist()
    pre = pre.tolist()
    for i in range(len(target)):
        if(pre[i] >= 0.03):
            if(target[i] >= 0.03):
                TP += 1
            else:
                FP += 1
        else:
            if(target[i] >= 0.03):
                TN += 1
            else:
                FN += 1
    return TP, FP, TN, FN

def accScore(TP, FP, TN, FN, a = 1):
    acc = (TP + FN)/(TP + FP + TN + FN)
    rec = (TP)/(TP + FN)
    pre = (TP)/(TP + FP)
    F = ((a * a + 1) * pre * rec)/(a * a * pre * rec)
    return acc, rec, pre, F

def testProcess(model, device, testData):
    '''
    参数:
    - model: 模型
    - device: 设备
    - train_loader: 训练数据集
    - optimizer: 优化器
    - epochs: 迭代次数
    - i: 折叠数
    - loss_fn: 损失值
    返回:
    - null
    '''
    model.eval()  # 将模型设置为评估模式
    with torch.no_grad():  # 在评估模式下遍历验证集数据
        for idx, (data, target) in enumerate(testData):
            data, target = data.to(device), target.to(device)  # 将数据和标签移动到指定设备上
            target = target.squeeze()  # 将目标的维度降为一维
            pre = model(data)  # 对数据运行模型，得到预测
            TP, FP, TN, FN = TPFN(target, pre)
            FScore = accScore(TP, FP, TN, FN)
            acc, rec, pre, F = FScore(TP, FP, TN, FN)
            print('Accuracy = ' + str(acc))
            print('Recall = ' + str(rec))
            print('Precision = ' + str(pre))
            print('F-Score = ' + str(F))
            return acc

if __name__ == '__main__':
    dataloader = fileProcess()
    dataload = np.squeeze(dataloader)  
    model = torch.load("resnet18.pth")
    device = setDevice()
    testData = dataload[2]
    testProcess(model, device, testData)