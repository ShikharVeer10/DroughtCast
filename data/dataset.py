from torch.utils.data import Dataset


class DroughtResearchDataset(Dataset):
    """Returns X, targets [scale, lead], valid-target mask, and sample metadata."""
    def __init__(self, X, Y, mask, metadata):
        self.X, self.Y, self.mask, self.metadata = X, Y, mask, metadata

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.mask[idx], self.metadata[idx]
