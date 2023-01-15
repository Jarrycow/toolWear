import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-data_dir', metavar='INPUT', type=str, default='./data/Data collection of machining pocket')  # 数据文件夹
    parser.add_argument('-fileLabel', metavar='INPUT', type=str, default='/Tool wear labels.xlsx')  # 标签文件
    return parser.parse_args()

def train_args():
    args = get_args()
    dataDir = args.data_dir  # 数据文件夹
    return dataDir

