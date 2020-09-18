from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import MNIST
import numpy as np
import random
import torch
import torchvision.transforms.functional as TF


class MNISTDataset(Dataset):

  def __init__(self, path, train, image_type, transform=None, download=True):
    self.mnist = MNIST(path, train, download=download)
    self.image_type = image_type
    self.transform = transforms.ToTensor() if transform is None else transform
    self.scales = np.arange(0.6, 1.6, 0.1)

  def __len__(self):
    return len(self.mnist)

  def __getitem__(self, idx):
    x, y = self.mnist[idx]
    if self.image_type == 0:
      return self.transform(x), y

    angle = random.randrange(-180, 180)
    tx, ty = random.randrange(9), random.randrange(9)
    scale = random.choice(self.scales)
    x_ = TF.affine(x, angle, (tx, ty), scale, shear=0)
    x_ = self.transform(x_)
    if self.image_type == 1:
      return x_, y

    x = self.transform(x)
    return torch.cat((x, x_)), torch.tensor((y, y))


class MNISTDataModule(LightningDataModule):

  def __init__(self, hparams):
    super().__init__()
    self.dataset_path = hparams.dataset_dir
    self.image_type = hparams.image_type
    self.kwargs = {'batch_size': hparams.batch_size, 
                   'num_workers': hparams.num_workers, 
                   'pin_memory': True}

  def prepare_data(self):
    # called only on 1 GPU
    transform = transforms.Compose([transforms.ToTensor()])
    self.train_mnist = MNISTDataset(
        self.dataset_path, 
        train=True, 
        image_type=self.image_type, 
        transform=transform, 
        download=True
    )
    self.val_mnist = MNISTDataset(
        self.dataset_path, 
        train=False, 
        image_type=self.image_type,
        transform=transform,
        download=True
    )

  # def setup(self):
  #   # called on every GPU
  #   vocab = load_vocab
  #   self.vocab_size = len(vocab)
  #   self.train, self.val, self.test = load_datasets()
  #   self.train_dims = self.train.next_batch.size()

  def train_dataloader(self):
    return DataLoader(self.train_mnist, shuffle=True, **self.kwargs)

  def val_dataloader(self):
    return DataLoader(self.val_mnist, shuffle=False, **self.kwargs)

  # def test_dataloader(self):
  #   transforms = ...
  #   return DataLoader(self.test, transforms)