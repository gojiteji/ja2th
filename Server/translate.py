# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import re
import time
import torch.optim as optimizers
import warnings
from flask import Flask,request
import nagisa
app = Flask(__name__)

#辞書クラス
class Vocabulary():
    def __init__(self,path):
        self.w2i={}
        self.i2w={}

        self.special_chars = ['<pad>','<s>','</s>','<unk>']
        self.bos_char = '<s>'
        self.eos_char = '</s>'
        self.oov_char = '<unk>'

        f = open(path)
        self.data = f.read()
        f.close()
        self.data = re.split('\n<EOS>\n',self.data) 
        for i in range(len(self.data)):
            tmp=re.split('\n', self.data[i] )
            self.data[i]=tmp[1:]#番号を削除
        self.data=self.data[:-1]#最後のデータいらない
        
        self._words=set()#値が被らない構造
        for datum in self.data:
            for i in range(len(datum)):
                self._words.update([datum[i]])

        
        self.w2i = {w: (i + len(self.special_chars))
                    for i, w in enumerate(self._words)}# i:index+3 w:word
        for i, w in enumerate(self.special_chars):# add 3
            self.w2i[w] = i
 
        self.i2w = {i: w for w, i in self.w2i.items()}
        
    def encode(self,words,teachershaping=False):
        output=[]
        #教師データ用の整形
        if teachershaping:
           words.append(self.eos_char)
           words.insert(0,self.bos_char)#前後に記号をつける
        for word in words:
            #辞書になし
            if word not in self.w2i:
                index = self.w2i[self.oov_char]#既存の<unk>を返す
            else:
            #辞書にあり
                index = self.w2i[word]#idを引っ張ってくる
            output.append(index)
        return output

    def decode(self,indexes):
        out=[]
        for index in indexes:
            out.append(self.i2w[index])
        return out



class ScaledDotProductAttention(nn.Module):
    def __init__(self,
                 d_k,
                 device='cuda'):
        super().__init__()
        self.device = device
        self.scaler = np.sqrt(d_k)

    def forward(self, q, k, v, mask=None):
        '''
        # Argument
            q, k, v: (batch, sequence, out_features)
            mask:    (batch, sequence)
        '''
        score = torch.einsum('ijk,ilk->ijl', (q, k)) / self.scaler
        score = score - torch.max(score, dim=-1, keepdim=True)[0]

        score = torch.exp(score)
        if mask is not None:
            if len(mask.size()) == 2:
                mask = mask.unsqueeze(1).repeat(1, score.size(1), 1)
            score.data.masked_fill_(mask, 0)

        a = score / torch.sum(score, dim=-1, keepdim=True)
        c = torch.einsum('ijk,ikl->ijl', (a, v))

        return c

class PositionalEncoding(nn.Module):
    def __init__(self, output_dim,
                 maxlen=6000,
                 device='cuda'):
        super().__init__()
        self.device = device
        self.output_dim = output_dim
        self.maxlen = maxlen
        pe = self.initializer()
        self.register_buffer('pe', pe)

    def forward(self, x, mask=None):
        pe = self.pe[:x.size(1), :].unsqueeze(0)
        
        return x + pe

    def initializer(self):
        pe = \
            np.array([[pos / np.power(10000, 2 * (i // 2) / self.output_dim)
                       for i in range(self.output_dim)]
                      for pos in range(self.maxlen)])

        pe[:, 0::2] = np.sin(pe[:, 0::2])
        pe[:, 1::2] = np.cos(pe[:, 1::2])

        return torch.from_numpy(pe).float()

class Attention(nn.Module):
    def __init__(self, output_dim, hidden_dim, device='cuda'):
        super().__init__()
        self.device = device
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.W_a = nn.Parameter(torch.Tensor(hidden_dim,
                                             hidden_dim))

        self.W_c = nn.Parameter(torch.Tensor(hidden_dim + hidden_dim,
                                             output_dim))

        self.b = nn.Parameter(torch.zeros(output_dim))

        nn.init.xavier_normal_(self.W_a)
        nn.init.xavier_normal_(self.W_c)

    def forward(self, ht, hs, source=None):
        '''
        # Argument
            ht, hs: (sequence, batch, out_features)
            source: (sequence, batch)
        '''
        score = torch.einsum('jik,kl->jil', (hs, self.W_a))
        score = torch.einsum('jik,lik->jil', (ht, score))

        score = score - torch.max(score, dim=-1, keepdim=True)[0]
        score = torch.exp(score)
        if source is not None:
            mask_source = source.t().eq(0).unsqueeze(0)
            score.data.masked_fill_(mask_source, 0)
        a = score / torch.sum(score, dim=-1, keepdim=True)

        c = torch.einsum('jik,kil->jil', (a, hs))
        h = torch.cat((c, ht), -1)
        return torch.tanh(torch.einsum('jik,kl->jil', (h, self.W_c)) + self.b)

class MultiHeadAttention(nn.Module):
    def __init__(self,h,d_model,device='cuda'):
        super().__init__()
        self.h = h
        self.d_model = d_model
        self.d_k = d_k = d_model // h
        self.d_v = d_v = d_model // h
        self.device = device

        self.W_q = nn.Parameter(torch.Tensor(h,
                                             d_model,
                                             d_k))

        self.W_k = nn.Parameter(torch.Tensor(h,
                                             d_model,
                                             d_k))

        self.W_v = nn.Parameter(torch.Tensor(h,
                                             d_model,
                                             d_v))

        nn.init.xavier_normal_(self.W_q)
        nn.init.xavier_normal_(self.W_k)
        nn.init.xavier_normal_(self.W_v)

        self.attn = ScaledDotProductAttention(d_k)
        self.linear = nn.Linear((h * d_v), d_model)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, q, k, v, mask=None):
        '''
        # Argument
            q, k, v: (batch, sequence, out_features)
            mask:    (batch, sequence)
        '''
        batch_size = q.size(0)

        q = torch.einsum('hijk,hkl->hijl',
                         (q.unsqueeze(0).repeat(self.h, 1, 1, 1),
                          self.W_q))
        k = torch.einsum('hijk,hkl->hijl',
                         (k.unsqueeze(0).repeat(self.h, 1, 1, 1),
                          self.W_k))
        v = torch.einsum('hijk,hkl->hijl',
                         (v.unsqueeze(0).repeat(self.h, 1, 1, 1),
                          self.W_v))

        q = q.view(-1, q.size(-2), q.size(-1))
        k = k.view(-1, k.size(-2), k.size(-1))
        v = v.view(-1, v.size(-2), v.size(-1))

        if mask is not None:
            multiples = [self.h] + [1] * (len(mask.size()) - 1)
            mask = mask.repeat(multiples)

        c = self.attn(q, k, v, mask=mask)
        c = torch.split(c, batch_size, dim=0)
        c = torch.cat(c, dim=-1)

        out = self.linear(c)

        return out


class Encoder(nn.Module):
    def __init__(self,depth_source,N=6,h=8,d_model=512,d_ff=2048,p_dropout=0.1,maxlen=128,device='cuda'):
        super().__init__()
        self.device = device
        self.embedding = nn.Embedding(depth_source,d_model, padding_idx=0)
        self.pe = PositionalEncoding(d_model, maxlen=maxlen)
        self.encoder_layers = nn.ModuleList([EncoderLayer(h=h,d_model=d_model,d_ff=d_ff,p_dropout=p_dropout,maxlen=maxlen,device=device) for _ in range(N)])
    def forward(self, x, mask=None):
        x = self.embedding(x)
        y = self.pe(x)
        for encoder_layer in self.encoder_layers:
            y = encoder_layer(y, mask=mask)
        return y
class EncoderLayer(nn.Module):
    def __init__(self,h=8,d_model=512,d_ff=2048,p_dropout=0.1,maxlen=128,device='cuda'):
        super().__init__()
        self.attn = MultiHeadAttention(h, d_model)
        self.dropout1 = nn.Dropout(p_dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FFN(d_model, d_ff)
        self.dropout2 = nn.Dropout(p_dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        h = self.attn(x, x, x, mask=mask)
        h = self.dropout1(h)
        h = self.norm1(x + h)
        y = self.ff(h)
        y = self.dropout2(y)
        y = self.norm2(h + y)
        return y


class Decoder(nn.Module):
    def __init__(self,depth_target,N=6,h=8,d_model=512,d_ff=2048,p_dropout=0.1,maxlen=128,device='cuda'):
        super().__init__()
        self.device = device
        self.embedding = nn.Embedding(depth_target,d_model, padding_idx=0)
        self.pe = PositionalEncoding(d_model, maxlen=maxlen)
        self.decoder_layers = nn.ModuleList([DecoderLayer(h=h,d_model=d_model,d_ff=d_ff,p_dropout=p_dropout,maxlen=maxlen,device=device) for _ in range(N)])
    def forward(self, x, hs,mask=None,mask_source=None):
        x = self.embedding(x)
        y = self.pe(x)
        for decoder_layer in self.decoder_layers:
            y = decoder_layer(y, hs,mask=mask,mask_source=mask_source)
        return y


class DecoderLayer(nn.Module):
    def __init__(self,h=8,d_model=512,d_ff=2048,p_dropout=0.1,maxlen=128,device='cuda'):
        super().__init__()
        self.self_attn = MultiHeadAttention(h, d_model)
        self.dropout1 = nn.Dropout(p_dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.src_tgt_attn = MultiHeadAttention(h, d_model)
        self.dropout2 = nn.Dropout(p_dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FFN(d_model, d_ff)
        self.dropout3 = nn.Dropout(p_dropout)
        self.norm3 = nn.LayerNorm(d_model)
    def forward(self, x, hs,mask=None,mask_source=None):
        h = self.self_attn(x, x, x, mask=mask)
        h = self.dropout1(h)
        h = self.norm1(x + h)
        z = self.src_tgt_attn(h, hs, hs,mask=mask_source)
        z = self.dropout2(z)
        z = self.norm2(h + z)
        y = self.ff(z)
        y = self.dropout3(y)
        y = self.norm3(z + y)
        return y

class FFN(nn.Module):
    def __init__(self, d_model, d_ff,
                 device='cpu'):
        super().__init__()
        self.l1 = nn.Linear(d_model, d_ff)
        self.a1 = nn.ReLU()
        self.l2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        h = self.l1(x)
        h = self.a1(h)
        y = self.l2(h)
        return y

class Transformer(nn.Module):
    def __init__(self,depth_source,depth_target,N=6,h=8,d_model=512,d_ff=2048,p_dropout=0.3,maxlen=128,device='cuda'):
        """
        (形態素数＝ベクトル数)
        depth_source:翻訳元の形態素数
        depth_target:翻訳後の形態素数
        N=6:エンコーダ/デコーダのレイヤを重ねる数
        h=8:Multi head attentionにける、attentionの数
        d_model=512:中間層の次元数
        d_ff=2048:フィードフォワード(通常のNNの順伝播)
        p_dropout=0.3:ドロップアウト率
        maxlen=128:形態素数の最大数

        """
        super().__init__()
        self.device = device
        self.encoder = Encoder(depth_source,N=N,h=h,d_model=d_model,d_ff=d_ff,p_dropout=p_dropout,maxlen=maxlen,device=device)
        self.decoder = Decoder(depth_target,N=N,h=h,d_model=d_model,d_ff=d_ff,p_dropout=p_dropout,maxlen=maxlen,device=device)
        self.out = nn.Linear(d_model, depth_target)
        nn.init.xavier_normal_(self.out.weight)

        self.maxlen = maxlen

    def forward(self, source, target=None):
        mask_source = self.sequence_mask(source)
        hs = self.encoder(source, mask=mask_source)

        if target is not None:
            target = target[:, :-1]
            len_target_sequences = target.size(1)
            mask_target = self.sequence_mask(target).unsqueeze(1)
            subsequent_mask = self.subsequence_mask(target)
            mask_target = torch.gt(mask_target + subsequent_mask, 0)
            y = self.decoder(target, hs,mask=mask_target,mask_source=mask_source)
            output = self.out(y)
        else:
            batch_size = source.size(0)
            len_target_sequences = self.maxlen
            output = torch.ones((batch_size, 1),dtype=torch.long,device=self.device)
            for t in range(len_target_sequences - 1):
                mask_target = self.subsequence_mask(output)
                out = self.decoder(output, hs,mask=mask_target,mask_source=mask_source)
                out = self.out(out)[:, -1:, :]
                out = out.max(-1)[1]
                output = torch.cat((output, out), dim=1)
        return output

    def sequence_mask(self, x):
        return x.eq(0)

    def subsequence_mask(self, x):
        shape = (x.size(1), x.size(1))
        mask = torch.triu(torch.ones(shape, dtype=torch.uint8),diagonal=1)
        return mask.unsqueeze(0).repeat(x.size(0), 1, 1).to(self.device)





"""
以下サーバー動作
"""


def translate_from_japanese(text):
    if(text is None):
        return "入力されていません"
    else:
        preds = model(torch.LongTensor([ja_vocabulary.encode(nagisa.tagging(text).words)]))
        _out = preds.view(-1).tolist()
        out = ' '.join(th_vocabulary.decode(_out))
        begin = out.find('<s>')
        end= out.find('</s>')
        return out[begin+3:end]


@app.route('/ja2th',methods=['POST'])
def ja2th():
    if request.method == 'POST':
        text = request.get_data()
    text=translate_from_japanese(text)
    return text

if __name__ == "__main__":
    warnings.simplefilter('ignore')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #辞書の作成
    ja_vocabulary=pd.read_pickle("ja.pkl") 
    th_vocabulary=pd.read_pickle("th.pkl") 
    
    depth_x = len(ja_vocabulary.i2w)
    depth_t = len(th_vocabulary.i2w)
    model = Transformer(depth_x,depth_t,N=7,h=3,d_model=32,d_ff=16,maxlen=21,device=device,p_dropout=0.1).to(device)
    model.load_state_dict(torch.load('7-3-32-16_90epoch.pt',map_location=device))

    app.run(debug=True)


