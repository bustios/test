import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import init


def init_weights(net, init_type='kaiming', init_gain=1.):
  """Initialize network weights.

  Args:
      net (network): network to be initialized
      init_type (str): the name of an initialization method: normal | xavier | kaiming | orthogonal
      init_gain (float): scaling factor for normal, xavier and orthogonal.
  """
  @torch.no_grad()
  def init_func(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
      if init_type == 'normal':
        init.normal_(m.weight, mean=0., std=1.)
      elif init_type == 'kaiming':
        init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
      elif init_type == 'xavier':
        init.xavier_normal_(m.weight.data, gain=init_gain)
      elif init_type == 'orthogonal':
        init.orthogonal_(m.weight, gain=init_gain)
      else:
        raise NotImplementedError(f'Initialization method {init_type} is not implemented')

  return net.apply(init_func)


class ComponentVAE(nn.Module):

  def __init__(self, in_channels=3, z_dim=16, full_res=False):
    super().__init__()
    self._in_channels = in_channels
    self._z_dim = z_dim
    # full res: 128x128, low res: 64x64
    h_dim = 4096 if full_res else 1024
    self.encoder = nn.Sequential(
      nn.Conv2d(in_channels + 1, 32, 3, stride=2, padding=1),
      nn.ReLU(True),
      nn.Conv2d(32, 32, 3, stride=2, padding=1),
      nn.ReLU(True),
      nn.Conv2d(32, 64, 3, stride=2, padding=1),
      nn.ReLU(True),
      nn.Conv2d(64, 64, 3, stride=2, padding=1),
      nn.ReLU(True),
      nn.Flatten(),
      nn.Linear(h_dim, 256),
      nn.ReLU(True),
      nn.Linear(256, 32)
    )
    self.decoder = nn.Sequential(
      nn.Conv2d(z_dim + 2, 32, 3),
      nn.ReLU(True),
      nn.Conv2d(32, 32, 3),
      nn.ReLU(True),
      nn.Conv2d(32, 32, 3),
      nn.ReLU(True),
      nn.Conv2d(32, 32, 3),
      nn.ReLU(True),
      nn.Conv2d(32, in_channels + 1, 1),
    )
    self._bg_logvar = 2 * torch.tensor(0.09).log()
    self._fg_logvar = 2 * torch.tensor(0.11).log()

  @staticmethod
  def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(mu)
    return mu + eps * std

  @staticmethod
  def spatial_broadcast(z, h, w):
    # Batch size
    n = z.shape[0]
    # Expand spatially: (n, z_dim) -> (n, z_dim, h, w)
    z_b = z.view(n, -1, 1, 1).expand(-1, -1, h, w)
    # Coordinate axes:
    x = torch.linspace(-1, 1, w, device=z.device)
    y = torch.linspace(-1, 1, h, device=z.device)
    x_b, y_b = torch.meshgrid(x, y)
    # Expand from (h, w) -> (n, 1, h, w)
    x_b = x_b.expand(n, 1, -1, -1)
    y_b = y_b.expand(n, 1, -1, -1)
    # Concatenate along the channel dimension: final shape = (n, z_dim + 2, h, w)
    z_sb = torch.cat((z_b, x_b, y_b), dim=1)
    return z_sb

  def forward(self, x, log_m_k, background=False):
    """
    :param x: Input image
    :param log_m_k: Attention mask logits
    :return: x_k and reconstructed mask logits
    """
    mu_logvar = self.encoder(torch.cat((x, log_m_k), dim=1))
    z_mu = mu_logvar[:, :self._z_dim]
    z_logvar = mu_logvar[:, self._z_dim:]
    z = self.reparameterize(z_mu, z_logvar) if self.training else z_mu

    # The height and width of the input to this CNN were both 8 larger than the
    # target output (i.e. image) size to arrive at the target size (i.e. 
    # accommodating for the lack of padding).
    h, w = x.shape[-2:]
    z_sb = self.spatial_broadcast(z, h + 8, w + 8)

    output = self.decoder(z_sb)
    x_mu = output[:, :self._in_channels]
    x_logvar = self._bg_logvar if background else self._fg_logvar
    m_logits = output[:, self._in_channels:]

    return m_logits, x_mu, x_logvar, z_mu, z_logvar


class AttentionBlock(nn.Module):

  def __init__(self, in_channels, out_channels, resize=True):
    super().__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
    self.norm = nn.InstanceNorm2d(out_channels, affine=True)
    self._resize = resize

  def forward(self, *inputs):
    downsampling = len(inputs) == 1
    x = inputs[0] if downsampling else torch.cat(inputs, dim=1)
    x = self.conv(x)
    x = self.norm(x)
    x = skip = F.relu(x, inplace=True)
    if self._resize:
      x = F.interpolate(skip, scale_factor=0.5 if downsampling else 2.,
                        mode='nearest', recompute_scale_factor=False)
    return (x, skip) if downsampling else x


class AttentionNetwork(nn.Module):
  """Unet-based network."""

  def __init__(self, in_channels=3, out_channels=1, n_filters=64):
    """Construct a Unet generator

    Parameters:
      in_channels (int): the number of channels in input images
      out_channels (int): the number of channels in output images
      n_filters (int): the number of filters in the first and last conv layers
    """
    super().__init__()
    self.downblock1 = AttentionBlock(in_channels + 1, n_filters)
    self.downblock2 = AttentionBlock(n_filters, n_filters * 2)
    self.downblock3 = AttentionBlock(n_filters * 2, n_filters * 4)
    self.downblock4 = AttentionBlock(n_filters * 4, n_filters * 8)
    self.downblock5 = AttentionBlock(n_filters * 8, n_filters * 8, resize=False)
    # self.downblock6 = AttentionBlock(n_filters * 8, n_filters * 8, resize=False)

    self.mlp = nn.Sequential(
      nn.Linear(4 * 4 * n_filters * 8, 128),
      nn.ReLU(inplace=True),
      nn.Linear(128, 128),
      nn.ReLU(inplace=True),
      nn.Linear(128, 4 * 4 * n_filters * 8),
      nn.ReLU(inplace=True)
    )

    # self.upblock1 = AttentionBlock(2 * n_filters * 8, n_filters * 8)
    self.upblock2 = AttentionBlock(2 * n_filters * 8, n_filters * 8)
    self.upblock3 = AttentionBlock(2 * n_filters * 8, n_filters * 4)
    self.upblock4 = AttentionBlock(2 * n_filters * 4, n_filters * 2)
    self.upblock5 = AttentionBlock(2 * n_filters * 2, n_filters)
    self.upblock6 = AttentionBlock(2 * n_filters, n_filters, resize=False)

    self.output = nn.Conv2d(n_filters, out_channels, 1)

  def forward(self, x, log_s_k):
    # Downsampling blocks
    x, skip1 = self.downblock1(torch.cat((x, log_s_k), dim=1))
    x, skip2 = self.downblock2(x)
    x, skip3 = self.downblock3(x)
    x, skip4 = self.downblock4(x)
    x, skip5 = self.downblock5(x)
    skip6 = skip5
    # The input to the MLP is the last skip tensor collected from the downsampling path (after flattening)
    # _, skip6 = self.downblock6(x)
    # Flatten
    x = skip6.flatten(start_dim=1)
    x = self.mlp(x)
    # Reshape to match shape of last skip tensor
    x = x.view(skip6.shape)
    # Upsampling blocks
    # x = self.upblock1(x, skip6)
    x = self.upblock2(x, skip5)
    x = self.upblock3(x, skip4)
    x = self.upblock4(x, skip3)
    x = self.upblock5(x, skip2)
    x = self.upblock6(x, skip1)
    # Output layer
    x = self.output(x)
    x = F.logsigmoid(x)
    return x


class SimpleBetaVAE(nn.Module):

  def __init__(self, hparams):
    super().__init__()
    self._z_dim = hparams.z_dim
    self.encoder = nn.Sequential(
      nn.Conv2d(hparams.input_channels, 32, 3, stride=1, padding=1),
      nn.ReLU(True),
      nn.Conv2d(32, 32, 3, stride=2, padding=1),
      nn.ReLU(True),
      nn.Conv2d(32, 64, 3, stride=1, padding=1),
      nn.ReLU(True),
      nn.Conv2d(64, 64, 3, stride=2, padding=1),
      nn.ReLU(True),
      nn.Flatten(),
      nn.Linear(7 * 7 * 64, self._z_dim * 2),
      nn.ReLU(True),
      nn.Linear(self._z_dim * 2, self._z_dim * 2)
    )
    self.decoder = SpatialBroadcastDecoder(hparams)

  @staticmethod
  def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(mu)
    return mu + eps * std

  def forward(self, x):
    mean_logvar = self.encoder(x)
    z_mean = mean_logvar[:, :self._z_dim]
    z_logvar = mean_logvar[:, self._z_dim:]
    z = self.reparameterize(z_mean, z_logvar) if self.training else z_mean
    x_tilde = self.decoder(z)

    output = dict(x_tilde=x_tilde, z_mean=z, z_logvar=z_logvar)
    return output


class SpatialBroadcastDecoder(nn.Module):

  def __init__(self, hparams):
    super().__init__()
    self._height = hparams.input_height
    self._width = hparams.input_width
    self.decoder = nn.Sequential(
      nn.Conv2d(hparams.z_dim + 2, 32, 3),
      nn.ReLU(True),
      nn.Conv2d(32, 32, 3),
      nn.ReLU(True),
      nn.Conv2d(32, 32, 3),
      nn.ReLU(True),
      nn.Conv2d(32, 32, 3),
      nn.ReLU(True),
      nn.Conv2d(32, hparams.output_channels, 1),
      nn.Sigmoid()
    )

  @staticmethod
  def spatial_broadcast(z, h, w):
    # Batch size
    n = z.shape[0]
    # Expand spatially: (n, z_dim) -> (n, z_dim, h, w)
    z_b = z.view(n, -1, 1, 1).expand(-1, -1, h, w)
    # Coordinate axes:
    x = torch.linspace(-1, 1, w, device=z.device)
    y = torch.linspace(-1, 1, h, device=z.device)
    x_b, y_b = torch.meshgrid(x, y)
    # Expand from (h, w) -> (n, 1, h, w)
    x_b = x_b.expand(n, 1, -1, -1)
    y_b = y_b.expand(n, 1, -1, -1)
    # Concatenate along the channel dimension, shape = (n, z_dim + 2, h, w)
    z_sb = torch.cat((z_b, x_b, y_b), dim=1)
    return z_sb

  def forward(self, z, h=None, w=None):
    if h is None:
      h = self._height
    if w is None:
      w = self._width
    
    z_sb = self.spatial_broadcast(z, h + 8, w + 8)
    x_tilde = self.decoder(z_sb)
    return x_tilde