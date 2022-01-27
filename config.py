# -*- encoding: utf-8 -*-
'''
@Describe:   
@File    :   config.py
@Time    :   2022/01/25 17:49:26
@Author  :   Victayria 
'''
from num_sequence import NumSequence

num_seq = NumSequence()
max_len = 9
train_batch_size = 256

embedding_dim = 100
num_layer = 1
hidden_size = 64

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_save_path = "./models/seq2seq.model"
optimizer_save_path = "./models/optimizer.model"
