import os
import torch
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class MultiDSpritesDataset(Dataset):

  def __init__(self, dataset_dir, dataset_size):
    self.dataset_dir = os.path.join(dataset_dir, 'multi_dsprites')
    self.dataset_size = dataset_size

  def __len__(self):
    return int(1000000 * self.dataset_size)

  def __getitem__(self, idx):
    filepath = os.path.join(self.dataset_dir, str(idx) + '.npz')
    data = np.load(filepath, allow_pickle=True)
    image = data['image']
    image = torch.from_numpy(image)
    # mask = data['mask']
    return image


class MultiDSpritesDataModule(LightningDataModule):

  def __init__(self, hparams):
    super().__init__()
    self.dataset_dir = hparams.dataset_dir
    self.dataset_size = hparams.dataset_size
    self.kwargs = {
        'batch_size': hparams.batch_size, 
        'num_workers': hparams.num_workers, 
        'pin_memory': True
    }

  def prepare_data(self):
    # called only on 1 GPU
    self.train_dataset = MultiDSpritesDataset(
        self.dataset_dir,
        self.dataset_size
    )

  def train_dataloader(self):
    return DataLoader(self.train_dataset, shuffle=True, **self.kwargs)

  # def val_dataloader(self):
  #   return DataLoader(self.val_dataset, shuffle=False, **self.kwargs)