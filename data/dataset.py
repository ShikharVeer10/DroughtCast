from torch.utils.data import Dataset


class DroughtResearchDataset(Dataset):
    """Returns X, targets [scale, lead], mask, metadata, and optional tensors.

    Optional keys (Phase 2 upgrades):
      last_spei [3]  — unshifted SPEI at the last input month (residual heads)
      static [S]     — region-agnostic descriptors (lat, aridity, …)
    """

    def __init__(self, X, Y, mask, metadata, last_spei=None, static=None):
        self.X, self.Y, self.mask, self.metadata = X, Y, mask, metadata
        self.last_spei = last_spei
        self.static = static

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        items = [self.X[idx], self.Y[idx], self.mask[idx], self.metadata[idx]]
        if self.last_spei is not None:
            items.append(self.last_spei[idx])
        if self.static is not None:
            items.append(self.static[idx])
        return tuple(items)
