# -*- encoding: utf-8 -*-
'''
@Describe:   实现解码器
@File    :   decoder.py
@Time    :   2022/01/26 16:23:53
@Author  :   Victayria 
'''
import torch.nn as nn
import torch.nn.functional as F
import torch
import config

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=len(config.num_seq), 
                                      embedding_dim=config.embedding_dim,
                                      padding_idx=config.num_seq.PAD)
        self.gru = nn.GRU(input_size=config.embedding_dim, 
                          hidden_size=config.hidden_size,
                          num_layers=config.num_layer,
                          batch_first=True)
        self.fc = nn.Linear(in_features=config.hidden_size, out_features=len(config.num_seq))
        
    def forward(self, target, encoder_hidden):
        # 1. 获取encoder的输出，作为decoder第一次的hidden_state
        decoder_hidden = encoder_hidden
        # 2. 准备decoder第一个时间步的输入，[batch_size, 1] SOS作为输入
        batch_size = target.size(0)
        decoder_input = torch.ones([batch_size, 1], dtype=torch.int64) * config.num_seq.SOS
        decoder_input = torch.LongTensor(decoder_input).to(config.device)
        # 3. 在第一个时间步上进行计算，得到第一个时间步的输出，hidden_state
        # 4. 把前一个时间步的输出进行计算，得到第一个最后的输出结果
        # 5. 把前一次的hidden_state作为当前时间步的hidden_state的输入，把前一次的输出，作为当前时间步的输入
        # 6. 循环4-5步骤
        
        # 保存预测结果
        decoder_outputs = torch.zeros([batch_size, config.max_len+1, len(config.num_seq)]).to(config.device)
        
        for t in range(config.max_len):
            # print("Time Step: {}".format(t))
            # decoder_input = decoder_input.to(torch.int64)
            decoder_output_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            # 保存decoder_ouput_t到decoder_outputs中
            decoder_outputs[:, t, :] = decoder_output_t
            value, index = torch.topk(decoder_output_t, 1) # 返回一个tuple(value, index)
            decoder_input = index
        return decoder_outputs, decoder_hidden
    
    def forward_step(self, decoder_input, decoder_hidden):
        """forward_step 计算每一个时间步上的结果
        Args:
            decoder_input ([batch_size, 1]): [description]
            decoder_hidden ([1, batch_size, hidden_size]): [description]
        Returns:
        """
        # print(decoder_input.size())
        # print(decoder_input.size())
        decoder_input_embedded = self.embedding(decoder_input) # [batch_size, 1, embedding_dim]
        
        # print(decoder_input_embedded.size())
        # print(decoder_hidden.size())
        out, decoder_hidden = self.gru(decoder_input_embedded, decoder_hidden)
        # out:[batch_size, 1, hidden_size]
        # decoder_hidden:[1, batch_size, hidden_size]
        
        out = out.squeeze(1)
        # print(out.size())
        # print(len(config.num_seq))
        output = self.fc(out) # [batch_size, vocab_size]  
        # print("forward_step: output.size(): " + str(output.size())) 
        output = F.log_softmax(output, dim=-1)
        # print("forward_step: docoder_hidden.size(): " + str(decoder_hidden.size()))
        # print(decoder_hidden)
        
        return output, decoder_hidden
    
    # 模型评估函数
    def eval(self, encoder_hidden):
        decoder_hidden = encoder_hidden # [1, batch_size, hidden_size]
        batch_size = encoder_hidden.size(1)
        decoder_input = torch.ones([batch_size, 1], dtype=torch.int64) * config.num_seq.SOS
        decoder_input = torch.LongTensor(decoder_input).to(config.device)
        
        indices = []
        
        # while True:
        for i in range(config.max_len + 2):
            decoder_output_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            value, index = torch.topk(decoder_output_t, 1)
            decoder_input = index
            # if index == config.num_seq.EOS:
            #     break
            # indices.append(index.item())
            indices.append(index.squeeze(-1))
        # print(indices)
        
        return indices
