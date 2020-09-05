

if __name__ == '__main__':
  from argparse import ArgumentParser
  from pprint import pprint

  from pytorch_lightning import Trainer
  from models.mnist_vae import MnistVAE
  from data_reader.mnist import MnistDataModule


  parser = ArgumentParser()
  parser = Trainer.add_argparse_args(parser)
  parser.add_argument('--batch_size', default=64, type=int)
  parser.add_argument('--dataset_path', default='./datasets', type=str)
  parser.add_argument('--learning_rate', default=1e-3, type=float)
  parser.add_argument('--num_workers', default=4, type=int)
  parser.add_argument('--vae_beta', default=0.5, type=float)
  parser.add_argument('--vae_z_dim', default=16, type=int)
  

  args = parser.parse_args()
  kwargs = {
    # 'fast_dev_run': True,
    'max_epochs': 10,
    # 'logger': False
  }

  dataloader = MnistDataModule(hparams=args)
  model = MnistVAE(beta=args.vae_beta, learning_rate=args.learning_rate)
  trainer = Trainer.from_argparse_args(args, **kwargs)
  trainer.fit(model, datamodule=dataloader)
  # pprint(vars(args))