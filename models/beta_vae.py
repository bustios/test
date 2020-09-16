"""C. P. Burgess et al., "MONet: Unsupervised Scene Decomposition and 
Representation," pp. 1–22, 2019."""

import argparse
import pytorch_lightning as pl
import torch
import torch.nn as nn

from .architectures import SimpleBetaVAE, init_weights


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
    self.model = init_weights(SimpleBetaVAE(hparams))
    self.eps = torch.finfo(torch.float).eps

  def forward(self, x):
    output = self.model(x)
    return output

  def loss_function(self, x_tilde, x, mean, logvar):
    BCE = nn.functional.binary_cross_entropy(x_tilde, x, reduction='sum')
    KLD = self.hparams.beta * torch.sum(logvar.exp() - logvar - 1 + mean.pow(2))
    return BCE + KLD, BCE, KLD

  def compute_loss(self, x):
    x = x.view(-1, 1, self.hparams.input_height, self.hparams.input_width)
    output = self(x)
    loss, recon_loss, kl_div = self.loss_function(
        output['x_tilde'], x, output['z_mean'], output['z_logvar'])
    return loss, recon_loss, kl_div, output

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), 
                                 lr=self.hparams.learning_rate)
    return optimizer

  def training_step(self, batch, batch_idx):
    x, _ = batch
    loss, recon_loss, kl_div, model_output = self.compute_loss(x)
    result = pl.TrainResult(loss)
    result.log_dict({
        'train_elbo_loss': loss,
        'train_recon_loss': recon_loss,
        'train_kl_loss': kl_div
    }, prog_bar=True)
    # self.logger.summary.scalar('loss', loss, step=self.global_step)
    return result

  def validation_step(self, batch, batch_idx):
    x, y = batch
    loss, recon_loss, kl_div, model_output = self.compute_loss(x)

    x = x.view(-1, 1, self.hparams.input_height, self.hparams.input_width)
    y = y.view(-1)

    tensorboard = self.logger.experiment
    tensorboard.add_embedding(model_output['z_mean'], y, x, global_step=self.current_epoch)
    # 'codes': ,
    #     'images': x,
    #     'labels': y
    result = pl.EvalResult(loss, checkpoint_on=loss)
    result.log_dict({
        'val_elbo_loss': loss,
        'val_recon_loss': recon_loss,
        'val_kl_div': kl_div
        
    })#, on_step=True)
    # log = {
    #     'val_elbo_loss': loss,
    #     'val_recon_loss': recon_loss,
    #     'val_kl_div': kl_div,
    #     'codes': model_output['z_mean'],
    #     'images': x,
    #     'labels': y
    # }
    # result = {'val_loss': loss, 'log': log}
    return result

  # def validation_epoch_end(self, val_step_outputs):
  #   loss = torch.stack([x['val_loss'] for x in val_step_outputs]).mean()
  #   codes = torch.cat([x['log']['codes'] for x in val_step_outputs])
  #   labels = torch.cat([x['log']['labels'] for x in val_step_outputs])
  #   images = torch.cat([x['log']['images']for x in val_step_outputs])
  #   tensorboard = self.logger.experiment
  #   tensorboard.add_embedding(codes, labels, images, self.current_epoch)
  #   return {'val_loss': loss}

  # def test_step(self, batch, batch_idx):
  #   x, y = batch
  #   y_hat = self(x)
  #   loss = F.cross_entropy(y_hat, y)
  #   return loss