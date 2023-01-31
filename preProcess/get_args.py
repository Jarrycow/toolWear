import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-data_dir', metavar='INPUT', type=str, default='./data/Data collection of machining pocket')  # 数据文件夹
    parser.add_argument('-fileLabel', metavar='INPUT', type=str, default='/Tool wear labels.xlsx')  # 标签文件
    parser.add_argument('-fold', metavar='INPUT', type=int, default=0)  # 折叠数
    parser.add_argument('-CUDA', metavar='INPUT', type=str, default='0')  # CUDA 版本
    parser.add_argument('-batch_size', metavar='INPUT', type=int, default=256)  # 批处理大小，每次迭代中使用的样本数
    parser.add_argument('-set_size', metavar='INPUT', type=int, default=5)  # 数据分割多少份
    parser.add_argument('-num_epochs', metavar='INPUT', type=int, default=100)  # 迭代次数
    parser.add_argument('-save_models', metavar='INPUT', type=str, default='./data/models/')  # 保存模型文件夹
    # TCN 参数
    parser.add_argument('-input_channel', metavar='INPUT', type=int, default=14)  # 输入通道数量
    parser.add_argument('-output_size', metavar='INPUT', type=int, default=4)  # 输出通道数量
    parser.add_argument('-in_ker_num', metavar='INPUT', type=int, default=64)  # 输入内核数量
    parser.add_argument('-dropout', metavar='INPUT', type=float, default=0.25)  # dropout
    parser.add_argument('-seq_leng',  metavar='INPUT',type=int, default=14)  # 序列长度
    parser.add_argument('-kernel_size',  metavar='INPUT',type=int, default=5)  # 内核大小
    parser.add_argument('-vocab_text_size',  metavar='INPUT',type=int, default=10)  # 语料库中词汇表的大小
    # CNN 参数
    parser.add_argument('-cnnConvKernel', metavar='INPUT', type=int, default=3)
    parser.add_argument('-cnnPoolKernel', metavar='INPUT', type=int, default=2)
    # BiGRU参数
    parser.add_argument('-layers', metavar='INPUT', type=int, default=4)  # 网络层数

    return parser.parse_args()

def train_args():
    args = get_args()
    dataDir = args.data_dir  # 数据文件夹
    return dataDir

