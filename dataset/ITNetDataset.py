from torch.utils import data
import torch.optim as optim
from PIL import Image
import cv2
import torch
from torchvision import transforms
import numpy as np
import os
class ITNetDataset(data.Dataset):
    '''
    该类是图片和目标文本
    '''
    def __init__(self,root_path,img_path,tgt_dict_path,tgt_path,max_token):
        
        self.root_path = root_path
        self.tgt_vocab= self.get_vocab(tgt_dict_path)#tgt词查token索引

        self.idx2word = {i: w for i, w in enumerate(self.tgt_vocab)}#token查词索引

        
        self.tgt_vocab_size = len(self.tgt_vocab)#tgt词表长度
        print('tgt词表大小：',self.tgt_vocab_size)

        self.src_img, self.tgt_sentence = self.get_word_line(img_path,tgt_path)#获取图像和目标语言句子
        self.tgt_token,self.output_token = self.make_data(self.tgt_sentence,max_token)
        print('src图像量：',len(self.src_img))
        print('tgt文本量：',len(self.tgt_sentence))




    
    def __getitem__(self,index):
        img_path = os.path.join(self.root_path,self.src_img[index]) 
        image = cv2.imread(img_path.replace('\n',''), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return torch.FloatTensor(image), self.tgt_token[index],self.output_token[index]
    
    def __len__(self):
        return len(self.tgt_vocab)
    
    def get_vocab(self,dict_path):
        '''
        加载字典
        这里的字典是使用fairseq脚本预处理过的字典
        key是词，value是token
        '''
        vocab_list = []
        
        with open(dict_path,'r',encoding='utf-8') as f:
            # print(dict_path,len(f.readlines()))
            for line in f.readlines():
                vocab_list.append(line.split()[0])
        
        dict_list = {}
        # out_index  = len(vocab_list)#有些词可能不在词表里，这里往后补充
        for i in range(len(vocab_list)):
                dict_list[vocab_list[i]] = i
        return dict_list


    def get_word_line(self,src_path,tgt_path):
        src_list = []
        tgt_list = []
        with open(src_path,'r',encoding='utf-8') as src_r:
            for src_line in src_r.readlines():
                src_list.append(src_line)
        with open(tgt_path,'r',encoding='utf-8') as tgt_r:
            for tgt_line in tgt_r.readlines():
                tgt_list.append(tgt_line)
        return src_list,tgt_list

    def make_data(self,tgt_list,max_token):
        dec_inputs, dec_outputs = [], []
        for tgt_line in tgt_list:
            dec_input = []
            for n in tgt_line.split():
                try:#可能词表里没有词
                    dec_input.append(self.tgt_vocab[n])
                except KeyError:
                    dec_input.append(1)#1代表未知
            # dec_input = [self.tgt_vocab[n] for n in tgt_line.split()]
            dec_output = []
            for n in tgt_line.split():
                try:#可能词表里没有词
                    dec_output.append(self.tgt_vocab[n])
                except KeyError:
                    dec_output.append(1)#1代表未知
            # dec_output = [self.tgt_vocab[n] for n in tgt_line.split()]
            dec_input = self.insert_pos(dec_input,max_token,1)
            dec_output = self.insert_pos(dec_output,max_token,2)
            dec_inputs.append(dec_input)
            dec_outputs.append(dec_output)
        return torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)


    def insert_pos(self,token_line,max_token,type = 0):
        '''
        处理数据，给不够长的数据加pad，给目标语言加start和end标识符
        type = 0默认是源语言
            1默认是加s
            2默认是加e
        max_token:token长度
        '''
        pad = 0
        S = 2
        E = 3
        if len(token_line) >= max_token:#先截断，按照最大的截断
            token_line = token_line[:max_token-1]
        if type == 0:#源语言，只需要padding
            token_line = fill_list(token_line,max_token,pad)#不够的填充pad
        elif type == 1:#目标输入，添加S
            token_line.insert(0,S)
            token_line = fill_list(token_line,max_token,pad)#不够的填充pad
        elif type == 2:#正确信息，添加E
            token_line.insert(-1,E)
            token_line = fill_list(token_line,max_token,pad)#不够的填充pad
        return token_line



def fill_list(my_list: list, length, fill=None): # 使用 fill字符/数字 填充，使得最后的长度为 length
    if len(my_list) >= length:
        return my_list
    else:
        return my_list + (length - len(my_list)) * [fill]