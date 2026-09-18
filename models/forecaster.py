from models.attention import TemporalAttention
import torch
import torch.nn as nn
from models.encoder import SharedLSTMEncoder
from models.decoder import ForecastDecoder

class MultiHorizonMultiHeadForecaster(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.encoder=SharedLSTMEncoder(
            input_dim=config['data']['input_dim'],
            hidden_dim_1=config['model']['hidden_dim_1'],
            hidden_dim_2=config['model']['hidden_dim_2'],
            dropout=config['model']['dropout']
        )

        self.attention=TemporalAttention(hidden_dim=config['model']['hidden_dim_2']) #Temporal Attention and Context Vector
        self.decoder_spei3 = ForecastDecoder(input_dim=config['model']['hidden_dim_2'], output_steps=config['data']['forecast_steps'])
        self.decoder_spei6 = ForecastDecoder(input_dim=config['model']['hidden_dim_2'], output_steps=config['data']['forecast_steps'])
        self.decoder_spei12 = ForecastDecoder(input_dim=config['model']['hidden_dim_2'], output_steps=config['data']['forecast_steps'])

    def forward(self,x):
        #Completion of forward pass through the multi-head framework

        encoder_out=self.encoder(x)
        context_vector,alpha=self.attention(encoder_out)

        prediction={
            'spei_3':self.decoder_spei3(context_vector),
            'spei_6':self.decoder_spei6(context_vector),
            'spei_12':self.decoder_spei12(context_vector)
        }
        return prediction,alpha