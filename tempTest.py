# from preProcess.fileProcess import A


import sys
print(sys.executable)
import torch
print(torch.__file__) 
print(torch.cuda.is_available())
from torch.utils import collect_env
print(collect_env.main())