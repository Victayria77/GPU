# -*- encoding: utf-8 -*-
'''
@Describe:   把encoder和decoder合并，得到seq2seq模型
@File    :   seq2seq.py
@Time    :   2022/01/26 17:19:44
@Author  :   Victayria 
'''
import torch.nn as nn
from decoder import Decoder
from encoder import Encoder


class Seq2Seq(nn.Module):
    def __init__(self):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def forward(self, input, target, input_len, target_len):
        encoder_outputs, encoder_hidden = self.encoder(input, input_len)
        decoder_outputs, decoder_hidden = self.decoder(target, encoder_hidden)
        return decoder_outputs, decoder_hidden
    
    def eval(self, input, input_len):
        encoder_outputs, encoder_hidden = self.encoder(input, input_len)
        indices = self.decoder.eval(encoder_hidden)
        return indices