from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

class MnistDataModule(LightningDataModule):

  def __init__(self, hparams):
    super().__init__()
    self.dataset_path = hparams.dataset_dir
    self.kwargs = {'batch_size': hparams.batch_size, 
                   'num_workers': hparams.num_workers, 
                   'pin_memory': True}

  def prepare_data(self):
    # called only on 1 GPU
    transform = transforms.Compose([transforms.ToTensor()])
    self.train_mnist = MNIST(self.dataset_path, train=True, download=True, 
                             transform=transform)
    self.val_mnist = MNIST(self.dataset_path, train=False, download=True,
                           transform=transform)

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