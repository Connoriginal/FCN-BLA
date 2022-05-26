from collections import OrderedDict
from turtle import forward

import torch
from torch import nn
from torch.nn.utils import rnn
import torch.nn.functional as F

class FCN(nn.Module):
    def __init__(self, image_dim = None):
        super(FCN, self).__init__()
        self.image_dim = image_dim
        # FCN layer
        self.fcn_blocks = nn.ModuleList()
        self.fcn_blocks.append(
            nn.Sequential(OrderedDict([
                ('Conv1_1', nn.Conv2d(3, 64, (3, 3), padding=1)),
                ('ReLU1_1', nn.ReLU()),
                ('Conv1_2', nn.Conv2d(64, 64, (3, 3), padding=1)),
                ('ReLU1_2', nn.ReLU()),
                ('MaxPool1', nn.MaxPool2d((2, 2))),
                ('Conv2_1', nn.Conv2d(64, 128, (3, 3), padding=1)),
                ('ReLU2_1', nn.ReLU()),
                ('Conv2_2', nn.Conv2d(128, 128, (3, 3), padding=1)),
                ('ReLU2_2', nn.ReLU()),
                ('MaxPool2', nn.MaxPool2d((2, 2))),
            ])))
        self.fcn_blocks.append(
            nn.Sequential(OrderedDict([
                ('Conv3_1', nn.Conv2d(128, 256, (3, 3), padding=1)),
                ('ReLU3_1', nn.ReLU()),
                ('Conv3_2', nn.Conv2d(256, 256, (3, 3), padding=1)),
                ('ReLU3_2', nn.ReLU()),
                ('Atrous1', nn.Conv2d(256, 256, (3, 3), dilation=2, padding=2)),
                ('ReLU_A1', nn.ReLU()),
            ])))
        self.fcn_blocks.append(
            nn.Sequential(OrderedDict([
                ('Conv4_1', nn.Conv2d(256, 256, (3, 3), padding=1)),
                ('ReLU4_1', nn.ReLU()),
                ('Conv4_2', nn.Conv2d(256, 256, (3, 3), padding=1)),
                ('ReLU4_2', nn.ReLU()),
                ('Atrous2', nn.Conv2d(256, 512, (3, 3), dilation=2, padding=2)),
                ('ReLU_A2', nn.ReLU()),
            ])))
        self.fcn_blocks.append(
            nn.Sequential(OrderedDict([
                ('Atrous3', nn.Conv2d(512, 512, (3, 3), dilation=2, padding=2)),
                ('ReLU_A3', nn.ReLU()),
                ('Atrous4', nn.Conv2d(512, 512, (3, 3), dilation=2, padding=2)),
                ('ReLU_A4', nn.ReLU()),
            ])))
        self.fcn_blocks.append(
            nn.Sequential(OrderedDict([
                ('Conv5', nn.Conv2d(1408, 512, (1, 1))),  # 1408 = 128 + 256 + 512 + 512 (hyper-atrous combination)
                ('ReLU5', nn.ReLU()),
                ('Deconv1', nn.ConvTranspose2d(512, 256, (3, 3), stride=2, padding=1, output_padding=1)),
                ('ReLU_D1', nn.ReLU()),
                ('Deconv2', nn.ConvTranspose2d(256, 64, (3, 3), stride=2, padding=1, output_padding=1)),
                ('ReLU_D2', nn.ReLU()),
                ('Conv6', nn.Conv2d(64, 1, (1, 1))),
            ])))
    
    def forward(self, X, mask=None):
        
        # X shape = N, L, C, H ,W 
        N, L, C, H, W = X.shape
        X = X.reshape(N*L,C,H,W)
        if mask is not None :
            mask = mask.reshape(N*L,1,H,W)
            X = X * mask
            
        h1 = self.fcn_blocks[0](X)
        h2 = self.fcn_blocks[1](h1)
        h3 = self.fcn_blocks[2](h2)
        h4 = self.fcn_blocks[3](h3)
        h = torch.cat((h1, h2, h3, h4), dim=1) # hyper-atrous combination
        h = self.fcn_blocks[4](h)
        density = h.reshape(N,L,1,H,W)
        if mask is not None :
            h = h * mask
        
        return density, h.sum(dim=(1,2,3)).reshape(N,L) # density & count
    
class Encoder(nn.Module):
    def __init__(self, image_dim = None):
        super(Encoder, self).__init__()
        self.image_dim = image_dim
        # Bidirectional LSTM layer
        H,W = self.image_dim
        # lstm, input size = H*W, hidden state size = 100 but bidirectional so double
        self.lstm_block = nn.LSTM(H*W,100,num_layers = 3, bidirectional = True, batch_first=True) 
        self.fc = nn.Linear(200,100) # (enc_hid_dim * 2, dec_hid_dim)
        
    def forward(self, density):
        # X shape = N, L, 1, H, W
        # count shape = N, L
        N,L,C,H,W = density.shape
        # count_FCN = density.sum(dim=(2,3,4)).reshape(N,L)
        
        h = density.reshape(N,L,-1)
        h, (hidden, cell) = self.lstm_block(h)
        # hidden [-2,:,:] = forward last hidden state, hidden[-1,:,:] = backward last hidden state
        
        return h, hidden, cell

class Decoder(nn.Module):
    def __init__(self, image_dim = None):
        super(Decoder, self).__init__()
        self.image_dim = image_dim
        H,W = self.image_dim
        self.lstm_block = nn.LSTM(H*W,100,num_layers = 3, bidirectional = True, batch_first=True)
        
        
    def forward(self, density, hidden, cell):
        # density shape = N, L, 1, H, W 
        # count shape = N, L
        N,L,C,H,W = density.shape
       
        h = density.reshape(N,L,-1)
        h, (hidden, cell) = self.lstm_block(h, (hidden, cell))
        
        return h, hidden, cell
       
        
class FCN_BLA(nn.Module):
    def __init__(self, FCN, Encoder, Decoder, image_dim = None):
        super(FCN_BLA, self).__init__()
        self.image_dim = image_dim
        self.FCN = FCN(image_dim = image_dim)
        self.Encoder = Encoder(image_dim = image_dim)
        self.Decoder = Decoder(image_dim = image_dim)

        self.W = nn.Linear(400,200)
        self.tanh = nn.Tanh()
        self.v = nn.Linear(200,1)
    
    def forward(self, X, mask):
        # X shape = N, L, C, H, W
        # mask shape = N, L
        density, count = self.FCN(X,mask)
        en_h, en_hidden, en_cell = self.Encoder(density[:,:-1])
        de_h, de_hidden, de_cell = self.Decoder(density[:,-1].unsqueeze(1), en_hidden, en_cell)
        # Add attention
        de_h_view = de_h.view(de_h.shape[0], de_h.shape[2], -1)
        score = torch.bmm(en_h, de_h_view)
        att_dis = F.softmax(score, dim=1)
        att_val = torch.sum(en_h * att_dis, dim=1)
        con = torch.cat((att_val, de_h.squeeze(1)), dim=1)
        out = self.v(self.tanh(self.W(con)))

        pred = count[:,-1].unsqueeze(1) + out
        return density, pred
        
    
        
        