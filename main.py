from torch.utils import data
from PIL import Image
from dataset.MyDataset import MyDataset
import torch.nn as nn
from model.transformer import Transformer
import torch.optim as optim
import os
import torch
from torch.optim.lr_scheduler import StepLR
train_dataset = MyDataset('data/data_bin_de_wmt14_img/dict.en.txt','data/data_bin_de_wmt14_img/dict.de.txt','data/data_bin_de_wmt14_img/train.en-de300k.en','data/data_bin_de_wmt14_img/train.en-de300k.de',50)
# print(img_dataset.__getitem__(3))
loader = data.DataLoader(train_dataset, batch_size=64, shuffle=True)
# img_dataloader = data.DataLoader(dataset=img_dataset, batch_size=16, shuffle=True)
# for batch_idx, (image, word_boxes) in enumerate(img_dataloader):
# 	print(batch_idx, (image.shape, word_boxes.shape))
#并没有什么新的事情发生
d_model = 256
d_k = 256
d_v = 256
n_heads = 16
d_ff = 2048
src_vocab_size = train_dataset.src_vocab_size
tgt_vocab_size = train_dataset.tgt_vocab_size
n_layers = 6


# print(src_vocab_size)
# print(tgt_vocab_size)

def save_point(epoch,model,optimizer,loss,PATH):
  torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, PATH
            )




model = Transformer(d_model,d_k,d_v,n_heads,d_ff,src_vocab_size,tgt_vocab_size,n_layers).cuda()
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=1.0,betas = (0.9,0.999), eps=1e-08,weight_decay = 0, amsgrad = False)
scheduler = StepLR(optimizer, step_size=2, gamma=0.1)

model.train()
for epoch in range(50):
    for enc_inputs, dec_inputs, dec_outputs in loader:
      '''
      enc_inputs: [batch_size, src_len]
      dec_inputs: [batch_size, tgt_len]
      dec_outputs: [batch_size, tgt_len]
      '''
      # enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(device), dec_inputs.to(device), dec_outputs.to(device)
      # outputs: [batch_size * tgt_len, tgt_vocab_size]
      enc_inputs, dec_inputs, dec_outputs = enc_inputs.cuda(), dec_inputs.cuda(), dec_outputs.cuda()
      outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
      
      loss = criterion(outputs, dec_outputs.view(-1))
      

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      scheduler.step()
    print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
    if (epoch+1) % 10 == 0:#十圈存一次
      save_point(epoch,model,optimizer,loss,'checkpoint/'+str(epoch+1)+'chekpoint.pt')


