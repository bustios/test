
if __name__ == '__main__':
  import torch
  from argparse import ArgumentParser
  from pytorch_lightning import Trainer

  from models.mnist_vae import MnistVAE
  from data_reader.mnist import MnistDataModule


  parser = ArgumentParser()
  parser = Trainer.add_argparse_args(parser)
  parser.add_argument(
    '--model_path', default='/content/drive/experiments/mnist_vae.pt', type=str)
  parser.add_argument('--output_dir', default='./logs', type=str)
  parser.add_argument('--dataset_path', default='./datasets', type=str)
  parser.add_argument('--num_workers', default=4, type=int)

  parser.add_argument('--batch_size', default=64, type=int)
  parser.add_argument('--learning_rate', default=1e-3, type=float)

  parser.add_argument('--in_channels', default=1, type=int)
  parser.add_argument('--out_channels', default=1, type=int)
  parser.add_argument('--height', default=28, type=int)
  parser.add_argument('--width', default=28, type=int)
  parser.add_argument('--beta', default=0.5, type=float)
  parser.add_argument('--z_dim', default=16, type=int)

  args = parser.parse_args('')

  dataloader = MnistDataModule(hparams=args)
  model = MnistVAE(hparams=args)
  trainer = Trainer.from_argparse_args(args)
  trainer.fit(model, datamodule=dataloader)

  torch.save(model.state_dict(), args.model_path)