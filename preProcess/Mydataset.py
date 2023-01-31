from torch.utils.data import Dataset
import torch
import  numpy as np
from preProcess.get_args import get_args
class Mydataset(Dataset):
    def __init__(self, x, y,seq_leng = get_args().seq_leng):  # 将数据转为Tensor格式
        super(Mydataset, self).__init__()
        x=np.array(x[:,:seq_leng])
        
        self.x_data = x.reshape(-1, seq_leng, 1)
        self.x_data = torch.FloatTensor(self.x_data)
        self.y_data = y.reshape(-1, 4, 1)
        self.y_data = torch.FloatTensor(self.y_data)
        pass

    def __len__(self):  # 
        return len(self.y_data)

    def __getitem__(self, idx):  # 
        return self.x_data[idx], self.y_data[idx]