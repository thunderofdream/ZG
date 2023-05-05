import torch.nn as nn
from model.encoder import Encoder
from model.decoder import Decoder
class Transformer(nn.Module):
    def __init__(self,d_model,d_k,d_v,n_heads,d_ff,src_vocab_size,tgt_vocab_size,n_layers):
        super(Transformer, self).__init__()
        self.encoder = Encoder(d_model,d_k,d_v,n_heads,d_ff,src_vocab_size,n_layers).cuda()
        self.decoder = Decoder(d_model,d_k,d_v,n_heads,d_ff,tgt_vocab_size,n_layers).cuda()
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False).cuda()
    def forward(self, enc_inputs, dec_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        '''
        # enc_outputs: [batch_size, src_len, d_model], enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        # dec_outpus: [batch_size, tgt_len, d_model], dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [n_layers, batch_size, tgt_len, src_len]
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs) # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns