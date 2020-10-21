import argparse
import itertools
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import optim

from .architectures import AttentionNetwork, ComponentVAE, init_net


class MONet(pl.LightningModule):

  def __init__(self, hparams):
    """Initialize this model class.

    Parameters:
        hparams: training/test hyperparameters
    """
    super().__init__()
    if isinstance(hparams, dict):
      hparams = argparse.Namespace(**hparams)

    self.save_hyperparameters(hparams)
    self.hparams = hparams
    self.attention_net = AttentionNetwork(hparams)
    self.comp_vae = ComponentVAE(hparams)
    self.eps = torch.finfo(torch.float).eps

  def init_parameters(self):
    self.attention_net = init_net(self.attention_net)
    self.comp_vae = init_net(self.comp_vae)

  def configure_optimizers(self):
    parameters = itertools.chain(
        self.attention_net.parameters(), 
        self.comp_vae.parameters()
    )
    optimizer = torch.optim.RMSprop(parameters, lr=self.hparams.learning_rate)
    return optimizer

  def compute_losses(self, x):
    encoder_loss = 0
    decoder_loss = []
    masks = []
    m_tilde_logits = []
    x_mu = []
    x_masked = []
    x_tilde = 0

    # Initial s_k = 1: shape = (N, 1, H, W)
    shape = list(x.shape)
    shape[1] = 1
    log_s_k = x.new_zeros(shape)

    for k in range(self.hparams.num_slots):
      # Derive mask from current scope
      if k != self.hparams.num_slots - 1:
        log_alpha_k = self.attention_net(x, log_s_k)
        log_m_k = log_s_k + log_alpha_k
        # Compute next scope
        log_s_k += (1. - log_alpha_k.exp()).clamp(min=self.eps).log()
      else:
        log_m_k = log_s_k

      # Get component and mask reconstruction, as well as the z_k parameters
      m_tilde_k_logits, x_mu_k, x_logvar_k, z_mu_k, z_logvar_k = self.comp_vae(x, log_m_k, k == 0)

      # KLD is additive for independent distributions
      encoder_loss += self.compute_kl_div(z_mu_k, z_logvar_k)

      # Exponents for the decoder loss
      loss_k = self.compute_reconstruction_loss(x, x_mu_k, x_logvar_k, log_m_k)
      decoder_loss.append(loss_k.unsqueeze(1))

      mask_k = log_m_k.exp()
      x_k_masked = mask_k * x_mu_k
      # # Iteratively reconstruct the output image
      # x_tilde += x_k_masked
      # Accumulate
      masks.append(mask_k)
      m_tilde_logits.append(m_tilde_k_logits)

      # x_mu.append(x_mu_k.unsqueeze(1))
      # x_masked.append(x_k_masked.unsqueeze(1))

    decoder_loss = torch.cat(decoder_loss, dim=1)
    masks = torch.cat(masks, dim=1)
    m_tilde_logits = torch.cat(m_tilde_logits, dim=1)
    
    # m_tilde = m_tilde_logits.softmax(dim=1)
    # x_mu = torch.cat(x_mu, dim=1)
    # x_masked = torch.cat(x_masked, dim=1)

    # return masks, m_tilde, x_mu, x_masked, x_tilde

    n = x.shape[0]
    encoder_loss /= n
    decoder_loss = -torch.logsumexp(decoder_loss, dim=1).sum() / n
    mask_loss = nn.functional.kl_div(m_tilde_logits.log_softmax(dim=1), masks, 
                                     reduction='batchmean')
    total_loss = self.hparams.beta * encoder_loss + decoder_loss \
                 + self.hparams.gamma * mask_loss

    return total_loss, encoder_loss, decoder_loss, mask_loss

  def compute_reconstruction_loss(self, x, x_mu, x_logvar, log_mask):
    loss = log_mask - 0.5 * x_logvar - (x - x_mu).pow(2) / (2 * x_logvar.exp())
    return loss

  def compute_kl_div(self, z_mean, z_logvar):
    kl_div = torch.sum(z_logvar.exp() - z_logvar - 1 + z_mean.pow(2))
    return kl_div

  def training_step(self, batch, batch_idx):
    x = batch
    total_loss, encoder_loss, decoder_loss, mask_loss = self.compute_losses(x)
    self.log_dict({
        'train_loss': total_loss,
        'train_encoder_loss': encoder_loss,
        'train_decoder_loss': decoder_loss,
        'train_mask_loss': mask_loss
    })
    return total_loss