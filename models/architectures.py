import torch
import torch.nn as nn
import torch.nn.functional as F


def init_net(net, init_type='trunc_normal', init_gain=0.02):
  """Initialize network weights.

  Args:
      net (network): network to be initialized.
      init_type (str): the name of an initialization method: trunc_normal | 
          normal | kaiming | xavier | orthogonal.
      init_gain (float): scaling factor for normal, xavier and orthogonal.
  """
  @torch.no_grad()
  def init_func(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
      if m.bias is not None:
        nn.init.constant_(m.bias, 0)
      if init_type == 'trunc_normal':
        nn.init.trunc_normal_(m.weight, mean=0., std=init_gain, a=-.1, b=.1)
      elif init_type == 'normal':
        nn.init.normal_(m.weight, mean=0., std=init_gain)
      elif init_type == 'kaiming':
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
      elif init_type == 'xavier':
        nn.init.xavier_normal_(m.weight, gain=init_gain)
      elif init_type == 'orthogonal':
        nn.init.orthogonal_(m.weight, gain=init_gain)
      else:
        raise NotImplementedError(f'Initialization method [{init_type}] is not implemented')

  print(f'Initialize network with {init_type}')
  return net.apply(init_func)


def compute_output_size(input_size, kernel_size, padding, stride, n_layers=1):
    output_size = input_size
    for _ in range(n_layers):
      output_size = 1 + (output_size - kernel_size + 2 * padding) // stride
    return output_size


class SpatialBroadcastDecoder(nn.Module):

  def __init__(self, hparams):
    super().__init__()
    self._height = hparams.input_height
    self._width = hparams.input_width
    self.decoder = nn.Sequential(
        nn.Conv2d(hparams.z_dim + 2, 32, 3),
        nn.ReLU6(True),
        nn.Conv2d(32, 32, 3),
        nn.ReLU6(True),
        nn.Conv2d(32, 32, 3),
        nn.ReLU6(True),
        nn.Conv2d(32, 32, 3),
        nn.ReLU6(True),
        nn.Conv2d(32, hparams.input_channels + 1, 1)
    )
    # Coordinate axes:
    # Gives the following error: RuntimeError: unsupported operation: more than
    # one element of the written-to tensor refers to a single memory location. 
    # Please clone() the tensor before performing the operation.
    # x = torch.linspace(-1, 1, self._width + 8)
    # y = torch.linspace(-1, 1, self._height + 8)
    # x_b, y_b = torch.meshgrid(x, y)
    # self.register_buffer('x_b', x_b)
    # self.register_buffer('y_b', y_b)

  def spatial_broadcast(self, z):
    # Batch size
    n = z.shape[0]
    # Expand spatially: (n, z_dim) -> (n, z_dim, h, w)
    z_b = z.view(n, -1, 1, 1).expand(-1, -1, self._height + 8, self._width + 8)
    # Coordinate axes:
    x = torch.linspace(-1, 1, self._width + 8)
    y = torch.linspace(-1, 1, self._height + 8)
    x_b, y_b = torch.meshgrid(x, y)
    # Expand from (h, w) -> (n, 1, h, w)
    x_b = x_b.expand(n, 1, -1, -1)
    y_b = y_b.expand(n, 1, -1, -1)
    # Concatenate along the channel dimension, shape = (n, z_dim + 2, h, w)
    z_sb = torch.cat((z_b, x_b, y_b), dim=1)
    return z_sb

  def forward(self, z):    
    z_sb = self.spatial_broadcast(z)
    output = self.decoder(z_sb)
    return output


class ComponentVAE(nn.Module):

  def __init__(self, hparams):
    super().__init__()
    height = compute_output_size(hparams.input_height, 3, 1, 2, 4)
    self._in_channels = hparams.input_channels
    self._z_dim = hparams.z_dim
    self.encoder = nn.Sequential(
        nn.Conv2d(self._in_channels + 1, 32, 3, stride=2, padding=1),
        nn.ReLU(True),
        nn.Conv2d(32, 32, 3, stride=2, padding=1),
        nn.ReLU(True),
        nn.Conv2d(32, 64, 3, stride=2, padding=1),
        nn.ReLU(True),
        nn.Conv2d(64, 64, 3, stride=2, padding=1),
        nn.ReLU(True),
        nn.Flatten(),
        nn.Linear(height * height * 64, 256),
        nn.ReLU(True),
        nn.Linear(256, self._z_dim * 2)
    )
    self.decoder = SpatialBroadcastDecoder(hparams)
    self._bg_logvar = 2 * torch.tensor(hparams.background_std).log()
    self._fg_logvar = 2 * torch.tensor(hparams.foreground_std).log()

  @staticmethod
  def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(mu)
    return mu + eps * std

  def encode(self, x, log_m_k):
    mu_logvar = self.encoder(torch.cat((x, log_m_k), dim=1))
    z_mu = mu_logvar[:, :self._z_dim]
    z_logvar = mu_logvar[:, self._z_dim:]
    return z_mu, z_logvar

  def forward(self, x, log_m_k, background=False):
    z_mu, z_logvar = self.encode(x, log_m_k)
    z = self.reparameterize(z_mu, z_logvar) if self.training else z_mu

    output = self.decoder(z)
    m_logits = output[:, self._in_channels:]
    x_mu = output[:, :self._in_channels]
    x_logvar = self._bg_logvar if background else self._fg_logvar
    
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

  def __init__(self, hparams, n_filters=64):
    super().__init__()
    self.downblock1 = AttentionBlock(hparams.input_channels + 1, n_filters)
    self.downblock2 = AttentionBlock(n_filters, n_filters * 2)
    self.downblock3 = AttentionBlock(n_filters * 2, n_filters * 4)
    self.downblock4 = AttentionBlock(n_filters * 4, n_filters * 8)
    self.downblock5 = AttentionBlock(n_filters * 8, n_filters * 8, resize=False)
    # self.downblock6 = AttentionBlock(n_filters * 8, n_filters * 8, resize=False)
    height = compute_output_size(hparams.input_height, 3, 1, 2, 4)
    self.mlp = nn.Sequential(
        nn.Linear(height * height * n_filters * 8, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, height * height * n_filters * 8),
        nn.ReLU(inplace=True)
    )

    # self.upblock1 = AttentionBlock(2 * n_filters * 8, n_filters * 8)
    self.upblock2 = AttentionBlock(2 * n_filters * 8, n_filters * 8)
    self.upblock3 = AttentionBlock(2 * n_filters * 8, n_filters * 4)
    self.upblock4 = AttentionBlock(2 * n_filters * 4, n_filters * 2)
    self.upblock5 = AttentionBlock(2 * n_filters * 2, n_filters)
    self.upblock6 = AttentionBlock(2 * n_filters, n_filters, resize=False)

    self.output = nn.Conv2d(n_filters, hparams.attention_out_channels, 1)

  def forward(self, x, log_s_k):
    # Downsampling blocks
    x, skip1 = self.downblock1(torch.cat((x, log_s_k), dim=1))
    x, skip2 = self.downblock2(x)
    x, skip3 = self.downblock3(x)
    x, skip4 = self.downblock4(x)
    x, skip5 = self.downblock5(x)
    skip6 = skip5
    # The input to the MLP is the last skip tensor collected from the
    # downsampling path (after flattening)
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