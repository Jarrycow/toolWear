from torch.utils.data import Dataset
import torch
import  numpy as np
class Mydataset(Dataset):
    def __init__(self, x, y):  # 将数据转为Tensor格式
        super(Mydataset, self).__init__()
        self.x_data = torch.LongTensor(x)
        self.y_data = torch.LongTensor(y)

    def __len__(self):  # 
        return len(self.y_data)

    def __getitem__(self, idx):  # 
        return self.x_data[idx], self.y_data[idx]