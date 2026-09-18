import torch
import torch.nn as nn

class ForcastDecoder(nn.Module):

    def __init__(self,input_dim=32,output_steps=4):
        super().__init__()
        self.decoder=nn.Sequential(nn.Linear(input_dim,32),nn.ReLU(),nn.Linear(32,output_steps))

    def forward(self,context_vector):
        return self.decoder(context_vector)