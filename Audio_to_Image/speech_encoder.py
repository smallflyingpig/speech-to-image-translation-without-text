import torch.nn as nn
import torch
import math
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F 
from functools import partial

def conv_layer_2d(in_channel, out_channel, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )

class CNNRNN(nn.Module):
    def __init__(self, n_filters, embedding_dim=1024, drop_prob=0.5,
                 nhidden=1024, nlayers=1, bidirectional=False, nsent=1024,):
        super(CNNRNN, self).__init__()
        self.bidirectional = bidirectional
        self.num_direction = 2 if bidirectional else 1
        self.embedding_dim = embedding_dim
        self.nhidden = nhidden // self.num_direction
        self.rnn_layers = nlayers
        self.drop_prob = drop_prob
        self.nsent = nsent // self.num_direction
        
        self.Conv = nn.Sequential(
            nn.BatchNorm2d(1),
            conv_layer_2d(1, 64, (n_filters, 1), (1,1), (0,0)), # 2048
            conv_layer_2d(64, 64, (1,3), (1,1), (0,1)), # 2048
            nn.MaxPool2d(kernel_size=(1,3), stride=(1,2), padding=(0,1)), # 1024
            conv_layer_2d(64, 128, (1,17), (1,2), (0,8)), # 512
            conv_layer_2d(128, 256, (1,13), (1,2), (0,6)), # 256
            conv_layer_2d(256, 256, (1,3), (1,1), (0,1)), # 256
            conv_layer_2d(256, 512, (1,9), (1,2), (0,4)), # 128
            nn.MaxPool2d(kernel_size=(1,3), stride=(1,2), padding=(0,1)), # 64
            conv_layer_2d(512, 512, (1,3), (1,1), (0,1)), # 64
            conv_layer_2d(512, 1024, (1,5), (1,2), (0,2)) # 32
        )
        self.RNN = nn.LSTM(
            self.embedding_dim, self.nhidden, num_layers=self.rnn_layers, batch_first=True, bidirectional=self.bidirectional, dropout=self.drop_prob
            )

        self.apply(self.weights_init)

    def init_hidden(self, x, n_dim):
        batch_size = x.shape[0]
        rtn = (torch.zeros(self.num_direction, batch_size, n_dim, device=x.device).requires_grad_(),
                torch.zeros(self.num_direction, batch_size, n_dim, device=x.device).requires_grad_())
        return rtn
        
    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.02)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def extract_feature(self, x, lens):
        words_emb, sent_emb = self.forward(x, lens)
        return sent_emb

    def forward(self, x, cap_lens):
        # (batch, channel, 40, 2048)
        # print(x.shape, x.device, next(self.parameters()).device)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        #print("before BN: shape: {}".format(x.shape))
        # print("before Conv: shape: {}".format(x.shape))
        x = self.Conv(x).squeeze()
        # print("after Conv: shape: {}".format(x.shape))
        x = x.transpose(1,2)
        cap_lens = (cap_lens).cpu().data.tolist()
        if isinstance(cap_lens, int):
            cap_lens = [cap_lens]
        batch_size, length = x.shape[:2]
        output = x

        h0 = self.init_hidden(x, self.nhidden)
        x = pack_padded_sequence(x, cap_lens, batch_first=True)
        output, hidden = self.RNN(x, h0)
        output = pad_packed_sequence(output, batch_first=True, total_length=length)[0]
        #print(output.shape)
        output = output.view(batch_size, length, self.rnn_layers, self.num_direction*self.nhidden)[:,:,-1,:]
        
        words_emb = output.transpose(1, 2)
        sent_emb = output.mean(-2) # hidden[0].transpose(0, 1).contiguous()
        sent_emb = sent_emb.view(-1, self.nsent * self.num_direction)
        # print(words_emb.shape, sent_emb.shape)
        # print(output.shape)
        return words_emb, sent_emb

