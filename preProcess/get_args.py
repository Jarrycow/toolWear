import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-data_dir', metavar='INPUT', type=str, default='./data/Data collection of machining pocket')  # 数据文件夹
    parser.add_argument('-fileLabel', metavar='INPUT', type=str, default='/Tool wear labels.xlsx')  # 标签文件
    parser.add_argument('-fold', metavar='INPUT', type=int, default=0)  # 折叠数
    parser.add_argument('-batch_size', metavar='INPUT', type=int, default=256)  # 批处理大小，每次迭代中使用的样本数
    parser.add_argument('-set_size', metavar='INPUT', type=int, default=5000)  # 数据分割多少份
    return parser.parse_args()

def train_args():
    args = get_args()
    dataDir = args.data_dir  # 数据文件夹
    return dataDir

