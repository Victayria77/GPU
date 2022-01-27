# -*- encoding: utf-8 -*-
'''
@Describe:   准备数据集，Dataset, DataLoader
@File    :   dataset.py
@Time    :   2022/01/25 17:38:47
@Author  :   Victayria 
'''
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import config

class NumDataset(Dataset):
    def __init__(self):
        # 使用numpy随机创建一堆数字
        import numpy as np
        np.random.seed(10) # 设置随机种子
        self.data = np.random.randint(0, 1e8, size=[500000]) # 造500000个数据
        
    
    def __getitem__(self, index):
        input = list(str(self.data[index])) # 将一个数字转换成字符串列表
        label = input + ["0"]
        input_length = len(input)
        label_length = len(label)
        return input, label, input_length, label_length
    
    def __len__(self):
        return self.data.shape[0]


def collate_fn(batch):
    '''
    :param batch: ([tokens, label, input_len, label_len], [tokens, label, input_len, label_len])一个getitem的结果
    :return 
    '''
    # 排序，按照长短排序
    batch = sorted(batch, key = lambda x: x[3], reverse=True) # 降序排序
    
    # content, label, input_lengths, label_lengths = list(zip(*batch))
    content, label, input_lengths, label_lengths = zip(*batch)
    content = [config.num_seq.transform(i, max_len=config.max_len) for i in content]
    label = [config.num_seq.transform(i, max_len=config.max_len + 1) for i in label]
    import torch
    content = torch.LongTensor(content)
    label = torch.LongTensor(label)
    input_lengths = torch.LongTensor(input_lengths)
    label_lengths = torch.LongTensor(label_lengths)
    
    return content, label, input_lengths, label_lengths

train_data_loader = DataLoader(NumDataset(), batch_size=config.train_batch_size, shuffle=False, collate_fn=collate_fn)
    
    
if __name__ == "__main__":
    num_dataset = NumDataset()
    print(num_dataset.data[:10])
    print(num_dataset[0])
    print(len(num_dataset))
    for input, label, input_len, label_len in train_data_loader:
        print(input)
        print(label)
        print("*"*50)
        print(input_len)
        print(label_len)
        # print(i.size)
        break