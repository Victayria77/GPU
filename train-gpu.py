from dataset import train_data_loader
from encoder import Encoder
from decoder import Decoder
from seq2seq import Seq2Seq
import torch.nn.functional as F
import config
import torch

# 训练流程
# 1. 实例化model, optimizer, loss
from torch.optim import Adam
seq2seq = Seq2Seq().to(config.device)
optimizer = Adam(seq2seq.parameters(), lr=0.01)

def train(epoch):
    # 2. 遍历dataloader
    from tqdm import tqdm
    # for index, (input, target, input_len, target_len) in tqdm(enumerate(train_data_loader)):
    bar = tqdm(enumerate(train_data_loader),
               total=len(train_data_loader), desc="train-gpu")
    for index, (input, target, input_len, target_len) in bar:
        input = input.to(config.device)
        target = target.to(config.device)
        input_len = input_len.to(config.device)
        target_len = target_len.to(config.device)
        optimizer.zero_grad()
        # 3. 调用得到output
        decoder_outputs, _ = seq2seq.forward(input, target, input_len, target_len)
        # print(decoder_outputs)
        # print("train: decoder_outputs.size():" + str(decoder_outputs.size()))
        # print("train: target.size():" + str(target.size()))
        # decoder_outputs = decoder_outputs.view(decoder_outputs.size(0)*decoder_outputs.size(1), -1)
        
        # 4. 计算loss
        decoder_outputs_ = decoder_outputs.view(-1, len(config.num_seq))
        target_ = target.view(-1)
        # print(type(target_))
        # print("train: decoder_outputs.size():" + str(decoder_outputs_.size()))
        # print("train: target.size():" + str(target_.size()))
        loss = F.nll_loss(decoder_outputs_, target_, ignore_index=config.num_seq.PAD)
        loss.backward() # 反向传播
        optimizer.step() # 参数更新
        
        # print(epoch, index, loss.item())
        
        bar.set_description(desc="epoch:{}, idx:{}, loss:{:.3f}".format(
            epoch, index, loss.item()), refresh=True)
        # 5. 模型保存和加载
        if index % 100 == 0:
            torch.save(seq2seq.state_dict(), config.model_save_path)
            torch.save(optimizer.state_dict(), config.optimizer_save_path)

if __name__ == "__main__":
    for i in range(10):
        train(i)
    # print(500000 / 128)