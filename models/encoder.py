import torch
import torch.nn as nn

class SharedLSTMEncoder(nn.Module):

    def __init__(self,input_dim=12,hidden_dim_1=64,hidden_dim_2=32,dropout=0.2):
        super().__init__()
        self.lstm1=nn.LSTM(input_dim,hidden_dim_1,batch_first=True)
        self.dropout=nn.Dropout(dropout)
        self.lstm2=nn.LSTM(hidden_dim_1,hidden_dim_2,batch_first=True)
    
    def forward(self,x):
        out, _=self.lstm1(x)
        out=self.dropout(out)
        out, _=self.lstm2(out)
        return out
