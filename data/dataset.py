import torch
from torch.utils.data import Dataset

class DroughtResearchDataset(Dataset):

    def __init__(self,X:torch.Tensor,y_3:torch.Tensor,y_6:torch.Tensor,y_12:torch.Tensor):
        self.X=X
        self.y_3=y_3
        self.y_6=y_6
        self.y_12=y_12

    
    def __len__(self): #How many total samples are in the dataset
        return len(self.X)

    #Grabs single historcial window and its corresponding multi-head targets by index 
    def __getitem__(self,idx):
        return self.X[idx], {
            'spei_3':self.y_3[idx],
            'spei_6':self.y_6[idx],
            'spei_12': self.y_12[idx]
        }
