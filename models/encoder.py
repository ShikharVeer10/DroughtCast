import torch
import torch.nn as nn

class SharedLSTMEncoder(nn.Module):

    def __init__(self,input_dim=12,hidden_dim_1=64,hidden_dim_2=32,dropout=0.2):
        super().__init__()
        #First LSTM layer capturing complex temporal patterns and dependencies
        self.lstm1=nn.LSTM(input_dim,hidden_dim_1,batch_first=True)
        self.dropout=nn.dropout(dropout)
        self.lstm2=nn.LSTM(hidden_dim_1,hidden_dim_2,batch_first=True)
    
    #Forward pass for the encoder where x(torch.Tensor) is the input tensor of shape [Batch. Seq_Len=6,Input_Dim=12]
    #This returns the torch.Tensor: Encoded in hidden states of shape[Batch,seq_len=6,Hidden_dim=32]
    def forward(self,x):
        out, _=self.lstm1(x)
        out=self.dropout(out)
        out, _=self.lstm2(out)
        return out
