

class NumSequence:
    PAD_TAG = "PAD"
    PAD = 0
    UNK_TAG = "UNK"
    UNK = 1
    SOS_TAG = "SOS" # start of sequence
    EOS_TAG = "EOS" # end of sequence
    SOS = 2
    EOS = 3
    
    
    def __init__(self):
        self.dict = {
            self.PAD_TAG : self.PAD,
            self.UNK_TAG : self.UNK,
            self.SOS_TAG : self.SOS,
            self.EOS_TAG : self.EOS
        }
        for i in range(10):
            self.dict[str(i)] = len(self.dict)
        
        self.inverse_dict = dict(zip(self.dict.values(), self.dict.keys()))
        
    def transform(self, sentence, max_len=9, add_eos=False): 
        """transform [把sentence转化为数字序列]
        Args:
            sentence ([type]): [description]
            add_eos (bool): [训练过程中，特征值需要加上EOS --> 长度变成max_len+1，目标值不需要加 --> 长度变成max_len]
        Returns:
        """        
        if len(sentence) > max_len:
            sentence = sentence[:max_len]
        raw_len = len(sentence)
        if add_eos:
            sentence += [self.EOS_TAG]
        if len(sentence) < max_len:
            sentence += [self.PAD_TAG] * (max_len - raw_len)
        res = [self.dict.get(i, self.UNK) for i in sentence]
        return res 

    def inverse_transform(self, indices):
        """inverse_transform 把序列转化回字符串
        Args:
            indices ([type]): [description]
        Returns:
        """
        # for i in indices:
        #     print(i)
        # print(self.inverse_dict)
        # res = [self.inverse_dict.get(i, self.UNK_TAG) for i in indices]
        res = []
        for idx, ind in enumerate(indices):
            tmp = self.inverse_dict.get(ind, self.UNK_TAG)
            # print("No.{}: {} --> {}".format(idx, ind, tmp))
            res.append(tmp)
        # print(res)
        return res
    
    
    def __len__(self):
        return len(self.dict)
    
    
if __name__ == "__main__":
    num_seq = NumSequence()
    print(num_seq.dict)