from torch.utils import data
import torch.optim as optim
from PIL import Image
# import cv2
import torch
from torchvision import transforms
import numpy as np
class MyDataset(data.Dataset):
    def __init__(self,src_dict_path,tgt_dict_path,src_path,tgt_path,max_token):
        self.src_vocab= self.get_vocab(src_dict_path)#src词查token索引
        self.tgt_vocab= self.get_vocab(tgt_dict_path)#tgt词查token索引

        self.idx2word = {i: w for i, w in enumerate(self.tgt_vocab)}#token查词索引

        self.src_vocab_size = len(self.src_vocab)#src词表长度
        self.tgt_vocab_size = len(self.tgt_vocab)#tgt词表长度
        print('src词表大小：',self.src_vocab_size)
        print('tgt词表大小：',self.tgt_vocab_size)

        self.src_sentence, self.tgt_sentence = self.get_word_line(src_path,tgt_path)#获取平行语料
        self.src_token,self.tgt_token,self.output_token = self.make_data(self.src_sentence, self.tgt_sentence,max_token)
        print('src文本量：',len(self.src_sentence))
        print('tgt文本量：',len(self.tgt_sentence))




    
    def __getitem__(self,index):
        return self.src_token[index], self.tgt_token[index],self.output_token[index]
    
    def __len__(self):
        return len(self.src_sentence)
    
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

    def make_data(self,src_list,tgt_list,max_token):
        enc_inputs, dec_inputs, dec_outputs = [], [], []
        for src_line,tgt_line in zip(src_list,tgt_list):
            enc_input = []
            for n in src_line.split():
                try:#可能词表里没有词
                    enc_input.append(self.src_vocab[n])
                except KeyError:
                    enc_input.append(1)#1代表未知
            # enc_input = [self.src_vocab[n] for n in src_line.split()]
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
            enc_input = self.insert_pos(enc_input,max_token)
            dec_input = self.insert_pos(dec_input,max_token,1)
            dec_output = self.insert_pos(dec_output,max_token,2)
            enc_inputs.append(enc_input)
            dec_inputs.append(dec_input)
            dec_outputs.append(dec_output)
        return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)


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

        # for i in range(len(sentences)):
        #     enc_input = [[self.src_vocab[n] for n in sentences[i][0].split()]] # [[1, 2, 3, 4, 0], [1, 2, 3, 5, 0]]
        #     dec_input = [[self.tgt_vocab[n] for n in sentences[i][1].split()]] # [[6, 1, 2, 3, 4, 8], [6, 1, 2, 3, 5, 8]]
        #     dec_output = [[self.tgt_vocab[n] for n in sentences[i][2].split()]] # [[1, 2, 3, 4, 8, 7], [1, 2, 3, 5, 8, 7]]

        #     enc_inputs.extend(enc_input)
        #     dec_inputs.extend(dec_input)
        #     dec_outputs.extend(dec_output)

        

    # def read_data(self,root_path,image_path):
    #     data_list = []
    #     with open(root_path+'/'+image_path, 'r', encoding='utf-8')as f_e:
    #         for i in f_e.readlines():
    #             data_list.append(i)
    #     return data_list
    
    # def get_img_and_box(self,index,root_path):
    #     text_split = self.data_list[index].split('###')
    #     img_name = text_split[0]
    #     image = cv2.imread(root_path+'/'+img_name, cv2.IMREAD_COLOR)
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     word_bboxes, words = self.load_img_gt_box(
    #         text_split[1:]
    #     )  # shape : (Number of word bbox, 4, 2)
    #     return np.array(image), word_bboxes
    # def load_img_gt_box(self, img_gt_box_text):
    #     box_split = img_gt_box_text
    #     word_bboxes = []
    #     words = []
    #     for item in box_split:
    #         item_split = item.split('##')
    #         words.append(item_split[-1])
    #         bboxes = eval(item_split[0])#框是以左上角为原点顺时针取
    #         box_points = np.array(bboxes, np.float32)
    #         word_bboxes.append(box_points)
    #     return np.array(word_bboxes), words