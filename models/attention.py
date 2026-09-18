import torch
import torch.nn as nn

class TemporalAttention(nn.Module):
    def __init__(self,hidden_dim=32):
        super().__init__() #Feed forward neural network to compute the importance scores for each time step
        self.attention_weights=nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.Tanh(),nn.Linear(hidden_dim,1))
        self.softmax=nn.Softmac(dim=1)

    def forward(self,encoder_outputs):
        #Forward pass for temporal attention

        scores=self.attention_weights(encoder_outputs) #Compute raw attention scores for each time step
        alpha=self.softmax(scores) #To apply softmax across the sequence dimension for the weights to sum to 1
        context_vector=torch.sum(alpha*encoder_outputs,dim=1)

        return context_vector,alpha