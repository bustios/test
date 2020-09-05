"""C. P. Burgess et al., "MONet: Unsupervised Scene Decomposition and 
Representation," pp. 1–22, 2019."""

import torch
import torch.nn as nn
from torch import optim

from .architectures import AttentionNetwork, ComponentVAE, init_net


class MONet(nn.Module):

  def __init__(self, opt):
    """Initialize this model class.

    Parameters:
        opt: training/test options

    A few things can be done here.
    - (required) call the initialization function of BaseModel
    - define loss function, visualization images, model names, and optimizers
    """
    super().__init__()
    self.opt = opt
    self.attention_net = init_net(AttentionNetwork())
    self.comp_vae = init_net(ComponentVAE())

    self.eps = torch.finfo(torch.float).eps
    self.criterionKL = nn.KLDivLoss(reduction='batchmean')

  def forward(self, x):
    """Run forward pass. This will be called by both functions 
    <optimize_parameters> and <test>."""
    self.x = x
    self.loss_E = 0
    x_tilde = 0
    nll = []
    m = []
    m_tilde_logits = []
    x_mu = []
    x_masked = []

    # Initial s_k = 1: shape = (N, 1, H, W)
    shape = list(self.x.shape)
    shape[1] = 1
    log_s_k = self.x.new_zeros(shape)

    for k in range(self.opt.num_slots):
      # Derive mask from current scope
      if k != self.opt.num_slots - 1:
        log_alpha_k = self.attention_net(self.x, log_s_k)
        log_m_k = log_s_k + log_alpha_k
        # Compute next scope
        log_s_k += (1. - log_alpha_k.exp()).clamp(min=self.eps).log()
      else:
        log_m_k = log_s_k

      # Get component and mask reconstruction, as well as the z_k parameters
      m_tilde_k_logits, x_mu_k, x_logvar_k, z_mu_k, z_logvar_k = self.comp_vae(x, log_m_k, k == 0)

      # KLD is additive for independent distributions
      self.loss_E += -0.5 * (1 + z_logvar_k - z_mu_k.pow(2) - z_logvar_k.exp()).sum()

      m_k = log_m_k.exp()
      x_k_masked = m_k * x_mu_k

      # Exponents for the decoder loss
      nll_k = log_m_k - 0.5 * x_logvar_k - (self.x - x_mu_k).pow(2) / (2 * x_logvar_k.exp())
      nll.append(nll_k.unsqueeze(1))

      # Get outputs for kth step
      # setattr(self, 'm{}'.format(k), m_k * 2. - 1.) # shift mask from [0, 1] to [-1, 1]
      # setattr(self, 'x{}'.format(k), x_mu_k)
      # setattr(self, 'xm{}'.format(k), x_k_masked)

      # Iteratively reconstruct the output image
      x_tilde += x_k_masked
      # Accumulate
      m.append(m_k)
      m_tilde_logits.append(m_tilde_k_logits)

      x_mu.append(x_mu_k.unsqueeze(1))
      x_masked.append(x_k_masked.unsqueeze(1))

    self.m = torch.cat(m, dim=1)
    self.m_tilde_logits = torch.cat(m_tilde_logits, dim=1)
    self.nll = torch.cat(nll, dim=1)

    m_tilde = self.m_tilde_logits.softmax(dim=1)
    x_mu = torch.cat(x_mu, dim=1)
    x_masked = torch.cat(x_masked, dim=1)
    return self.m, m_tilde, x_mu, x_masked, x_tilde

  def backward(self):
    """Calculate losses, gradients, and update network weights; called in every
    training iteration"""
    n = self.x.size(0)
    self.loss_E /= n
    self.loss_D = -torch.logsumexp(self.nll, dim=1).sum() / n
    self.loss_mask = self.criterionKL(self.m_tilde_logits.log_softmax(dim=1), self.m)
    loss = self.loss_D + self.opt.beta * self.loss_E + self.opt.gamma * self.loss_mask
    loss.backward()