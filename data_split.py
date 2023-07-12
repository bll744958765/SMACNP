#coding=gbk
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random
import torch
import os
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

set_seed(seed=42)
data_frame = pd.read_csv(r'./generation_data_noise.csv')
all_data = np.array(data_frame)

for i in range(10):
    train_data, valid_data = train_test_split(all_data, test_size=0.7)
    train_data=pd.DataFrame(train_data)
    valid_data=pd.DataFrame(valid_data)
    print(len(train_data),len(valid_data))
    train_data.to_csv('./train{}.csv'.format(i), index=False)
    valid_data.to_csv('./valid{}.csv'.format(i), index=False)
