# -*- encoding: utf-8 -*-
'''
@Describe:   编码器
@File    :   encoder.py
@Time    :   2022/01/25 20:57:56
@Author  :   Victayria 
'''
import torch.nn as nn
import config
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=len(config.num_seq), 
                                      embedding_dim=config.embedding_dim,
                                      padding_idx=config.num_seq.PAD) # padding_idx 明确告诉哪些值为padding值，不需要进行更新
        self.gru = nn.GRU(input_size=config.embedding_dim, 
                          num_layers=config.num_layer,
                          hidden_size=config.hidden_size,
                          batch_first=True)
        
    def forward(self, input, input_len, batch_first=True):
        # input:[batch_size, max_len]
        embeded = self.embedding(input) # [batch_size, max_len, embedding_dim]
        # input_len = input_len.cpu()
        embeded = pack_padded_sequence(embeded, lengths=input_len.cpu(), batch_first=batch_first) # 打包
        output, hidden = self.gru(embeded)
        # 解包
        out, out_len = pad_packed_sequence(output, batch_first=True, padding_value=config.num_seq.PAD)
        
        # hidden:[num_layer * bi, batch_size, hidden_size] = [1*1, batch_size, hidden_size]
        # out:[batch_size, seq_len, hidden_size]
        return out, hidden

if __name__ == "__main__":
    from dataset import train_data_loader
    encoder = Encoder()
    for input, target, input_len, target_len in train_data_loader:
        out, hidden, out_len = encoder(input, input_len=input_len)
        print(out.size())
        print(hidden.size())
        print(out_len)
        break