# -*- encoding: utf-8 -*-
'''
@Describe:   完成模型的评估
@File    :   eval.py
@Time    :   2022/01/26 19:30:30
@Author  :   Victayria 
'''
from dataset import train_data_loader
from encoder import Encoder
from decoder import Decoder
from seq2seq import Seq2Seq
import torch.nn.functional as F
import config
import torch
import numpy as np




# 1. 准备测试数据集

data = [str(i) for i in np.random.randint(0, 1e8, size=[100])] # input
data = sorted(data, key=lambda x:len(x), reverse=True)

target = [i + '0' for i in data]

# print(data)
input_len = torch.LongTensor([len(i) for i in data]).to(config.device)
input = torch.LongTensor([config.num_seq.transform(list(sentence), config.max_len) for sentence in data]).to(config.device)

# 2. 实例化模型，加载model
seq2seq = Seq2Seq()
seq2seq = seq2seq.to(config.device)
seq2seq.load_state_dict(torch.load(config.model_save_path))

# 3. 获取预测值
indices = seq2seq.eval(input, input_len)
# indices = np.asarray([i.cpu().detach().numpy() for i in indices])
# # print(indices)
# indices = indices.transpose()
indices = np.array(indices).transpose()
# print(indices)
# 4. 反序列化，观察结果
# res = [config.num_seq.inverse_transform(i) for i in indices]
res = []
for line in indices:
    tmp = config.num_seq.inverse_transform(line)
    cur_line = ""
    for word in tmp:
        if word == config.num_seq.EOS_TAG:
            break
        cur_line += word
    res.append(cur_line)
print(data[:10])
print(res[:10])

print(target[:10])
print(sum([ i==j for i,j in zip(target, res)]) / len(target))