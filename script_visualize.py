import torch
import pytorch_lightning as pl
import args_parser
from models.beta_vae import BetaVAE
from data_reader.mnist import MNISTDataModule
from visualize.visualize_model import visualize


def get_ordered_batch(val_dataloader):
  it = iter(val_dataloader)
  _, _ = next(it)
  x, y = next(it)
  batch = []
  for i in 23, 5, 3, 0, 10, 13, 18, 2, 29, 26:
    batch.append(x[i])

  return torch.stack(batch)


def visualize_outputs():
  pl.seed_everything(0)
  parser = args_parser.get_parser()
  parser.add_argument('checkpoint_path', type=str)
  parser.add_argument('--output_dir', default='./logs', type=str)
  args = parser.parse_args()
  dataloader = MNISTDataModule(hparams=args)
  dataloader.prepare_data()
  val_dataloader = dataloader.val_dataloader()
  x = get_ordered_batch(val_dataloader)

  model = BetaVAE.load_from_checkpoint(args.checkpoint_path)
  decoder = model.model.decoder
  visualize(model, decoder, x, args.output_dir, latent_dims=args.z_dim)


if __name__ == '__main__':
  visualize_outputs()