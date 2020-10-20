import argparse
import pytorch_lightning as pl
import torch
import torch.nn as nn

from .architectures import SimpleBetaVAE, init_net


class BetaVAE(pl.LightningModule):

  def __init__(self, hparams):
    """Initialize this model class.

    Parameters:
        hparams: training/test hyperparameters
    """
    super().__init__()
    if isinstance(hparams, dict):
      hparams = argparse.Namespace(**hparams)
    # Anything assigned to self.hparams will be saved automatically
    self.hparams = hparams
    self.model = SimpleBetaVAE(hparams)
    self.eps = torch.finfo(torch.float).eps
    self.codes = []
    self.images = []
    self.labels = []

  def init_parameters(self):
    self.model = init_net(self.model)

  def forward(self, x):
    output = self.model(x)
    return output

  def compute_loss(self, x, x_tilde, z_mean, z_logvar):
    bce = nn.functional.l1_loss(x_tilde, x, reduction='sum')
    kl_div = self.hparams.beta * torch.sum(z_logvar.exp() - z_logvar - 1 
                                           + z_mean.pow(2))
    loss = bce + kl_div
    return loss, bce, kl_div

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), 
                                 lr=self.hparams.learning_rate)
    return optimizer

  def training_step(self, batch, batch_idx):
    x, _ = batch
    x = x.view(-1, 1, self.hparams.input_height, self.hparams.input_width)
    output = self.model(x)
    loss, reconstruction_loss, kl_div = self.compute_loss(
        x, output['x_tilde'], output['z_mean'], output['z_logvar'])

    self.log_dict({
        'train_elbo_loss': loss,
        'train_reconstruction_loss': reconstruction_loss,
        'train_kl_loss': kl_div
    })

    return loss

  def validation_step(self, batch, batch_idx):
    x, y = batch
    x = x.view(-1, 1, self.hparams.input_height, self.hparams.input_width)
    output = self.model(x)
    loss, reconstruction_loss, kl_div = self.compute_loss(
        x, output['x_tilde'], output['z_mean'], output['z_logvar'])
    if self.current_epoch == self.trainer.max_epochs - 1:
      self.codes.append(output['z_mean'])
      self.labels.extend(y.view(-1).tolist())
      self.images.append(x)

    self.log_dict({
        'val_loss': loss,
        'val_reconstruction_loss': reconstruction_loss,
        'val_kl_div': kl_div
    })

    return loss

  # def validation_epoch_end(self, val_step_outputs):
    # pass

  # def test_step(self, batch, batch_idx):
  #   x, y = batch
  #   y_hat = self(x)
  #   loss = F.cross_entropy(y_hat, y)
  #   return loss