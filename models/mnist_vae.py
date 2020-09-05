"""C. P. Burgess et al., "MONet: Unsupervised Scene Decomposition and 
Representation," pp. 1–22, 2019."""

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision

from .architectures import SimpleBetaVAE, init_weights


class MnistVAE(pl.LightningModule):

  def __init__(self, beta=0.5, learning_rate=1e-3):
    """Initialize this model class.

    Parameters:
        hparams: training/test hyperparameters
    """
    super().__init__()
    self.beta = beta
    self.learning_rate = learning_rate
    self.model = init_weights(SimpleBetaVAE())
    self.eps = torch.finfo(torch.float).eps

  def forward(self, x):
    output = self.model(x)
    return output

  def loss_function(self, x_tilde, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(x_tilde, x, reduction='sum')
    KLD = self.beta * torch.sum(logvar.exp() - logvar - 1 + mu.pow(2))
    return BCE + KLD

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    return optimizer

  def training_step(self, batch, batch_idx):
    x, _ = batch
    output = self.model(x)
    loss = self.loss_function(output['x_tilde'], x, output['mu'], output['logvar'])
    return {'loss': loss}

  def validation_step(self, batch, batch_idx):
    x, _ = batch
    output = self.model(x)
    loss = self.loss_function(output['x_tilde'], x, output['mu'], output['logvar'])
    return {'val_loss': loss}

  def validation_epoch_end(self, val_step_outputs):
    val_loss = torch.stack([x['val_loss'] for x in val_step_outputs]).mean()
    log = {'avg_val_loss': val_loss}
    return {'val_loss': val_loss, 'log': log}

  # def test_step(self, batch, batch_idx):
  #   x, y = batch
  #   y_hat = self(x)
  #   loss = F.cross_entropy(y_hat, y)
  #   return loss