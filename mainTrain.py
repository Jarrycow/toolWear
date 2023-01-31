
import os
import numpy as np
from sklearn.metrics import balanced_accuracy_score

import torch
from torch.optim import lr_scheduler
from mainTest import testProcess

from preProcess.get_args import get_args

from preProcess.fileProcess import fileProcess

from model.TCN import TCN
from model.BiGRU import BiGRU
from model.CNN import CNN

def setDevice(): # 设置设备
    args = get_args()
    CUDA_VISIBLE_DEVICES = args.CUDA  # CUDA版本
    os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES  # 设置CUDA
    device = torch.device("cuda"if torch.cuda.is_available() else "cpu")  # 调用GPU
    device = torch.device("cpu")
    return device

def train(model, device, train_loader, optimizer, epochs, i, loss_fn, trainModel):  # 
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
    args = get_args()
    path = args.save_models + trainModel + '.pth'
    
    model.train()  # 启用训练模式
    total_loss = 0  # 初始化总损失
    sum_num = 0.   # 初始化总样本数
    for idx, (data, target) in enumerate(train_loader):  # 遍历训练数据
        print(idx)
        data, target = data.to(device), target.to(device)  # 将数据和标签复制到指定设备
        target = target.squeeze()  # 去除标签的维度
        pre = model(data)  # 使用模型预测
        loss = loss_fn(pre.to(device), target.to(device).float()).to(device)  # 计算损失
        optimizer.zero_grad()  # 清空模型梯度
        loss.backward()  # 计算梯度
        optimizer.step()  # 更新模型参数
        
        total_loss += loss.item() * len(target)  # 累加损失
        
        sum_num += len(target)  # 累加样本数
        if idx % 100 == 99:  # 每进行100次训练，打印训练信息
            print("Fold {} Train Epoch: {}, iteration: {}, Loss: {}".format(i, epochs, idx + 1, loss.item()))
    torch.save(model.state_dict(), path)
    print("---------------------------------------------------")  # 打印训练信息

def trainProcess(dataload, trainModel):  # 训练
    '''
    参数: 
    - train_x: 训练集自变量
    - train_y: 训练集因变量
    - train_model: 训练模型
    返回: 
    '''
    args = get_args()
    epochs = args.num_epochs  # 迭代次数
    fold = args.fold  # 折叠数
    
    
    trainData = dataload[0]
    validData = dataload[1]

    device = setDevice()
    if trainModel == 'TCN':
        model = modelTCN(device)
    elif trainModel == 'GRU':
        model = modelGRU(device)
    elif trainModel == 'CNN':
        model = modelCNN(device)
    elif trainModel == 'CNN_BiGRU':
        model = modelCNN_BiGRU(device)
    else:  
        return

    criterion = torch.nn.CrossEntropyLoss()  # 损失函数
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=5e-3, weight_decay=0)  # 使用默认参数的Adam优化器，权重衰减为0之外
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=20, verbose=True, min_lr=1e-5)  # 如果验证准确率没有提高，使用调度器来降低学习率
    max_acc = 0  # 掌握最大验证精度
    for epoch in range(epochs):  # 对模型进行指定数量的迭代训练
        train(model, device, trainData, optimizer, epoch, int(fold)+1, criterion, trainModel)  # 训练
        valid_acc = testProcess(model, device, validData)  # 在验证集上评估模型，并根据验证精度保存最佳模型
        if max_acc < valid_acc:  #
            max_acc = valid_acc
            
def modelTCN(device):
    args = get_args()
    input_channel = args.input_channel
    output_size = args.output_size
    num_channels = [args.in_ker_num] * args.layers  #  每一层内核数量
    kernel_size = args.kernel_size
    dropout = args.dropout
    vocab_text_size = args.vocab_text_size
    seq_leng = args.seq_leng
    

    model = TCN(input_channel=input_channel, output_size=output_size, num_channels=num_channels, 
                kernel_size=kernel_size, dropout=dropout,
                vocab_text_size=vocab_text_size, seq_leng=seq_leng).to(device)  # 创造TCN模型
    return model

def modelGRU(device):
    args = get_args()
    input_channel = args.input_channel
    output_size = args.output_size
    kernel_size = args.kernel_size
    layers = args.layers
    model = BiGRU(input_channel, kernel_size, layers, output_size).to(device)  # 创造GRU模型
    return model

def modelCNN(device):
    args = get_args()
    input_channel = args.input_channel
    output_size = args.output_size
    cnnConvKernel = args.cnnConvKernel
    cnnPoolKernel = args.cnnPoolKernel

    model = CNN(input_channel, output_size, cnnConvKernel, cnnPoolKernel).to(device)
    return model

def modelCNN_BiGRU(device):
    args = get_args()
    input_channel = args.input_channel
    output_size = args.output_size
    num_channels = args.layers
    kernel_size = args.cnnConvKernel
    dropout = args.dropout
    vocab_text_size = args.vocab_text_size
    seq_leng = args.seq_leng
    model = TCN(input_channel, output_size, num_channels, kernel_size, dropout, vocab_text_size, seq_leng).to(device)
    return model


if __name__ == '__main__':
    dataloader = fileProcess()
    dataload = np.squeeze(dataloader)  
    print('Data Load!')
    trainProcess(dataload,'TCN')
    # trainProcess(dataload,'GRU')
    # trainProcess(dataload, 'CNN')
    trainProcess(dataload, 'CNN-BiGRU')

    
    
    