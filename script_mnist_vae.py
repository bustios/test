

if __name__ == '__main__':
  from argparse import ArgumentParser
  from pprint import pprint
  from pytorch_lightning import Trainer

  from models.mnist_vae import MnistVAE
  from data_reader.mnist import MnistDataModule


  parser = ArgumentParser()
  parser = Trainer.add_argparse_args(parser)
  parser.add_argument('--dataset_path', default='./datasets', type=str)
  parser.add_argument('--num_workers', default=4, type=int)

  parser.add_argument('--batch_size', default=64, type=int)
  parser.add_argument('--learning_rate', default=1e-3, type=float)
  
  parser.add_argument('--vae_beta', default=0.5, type=float)
  parser.add_argument('--vae_z_dim', default=16, type=int)
  

  args = parser.parse_args()

  dataloader = MnistDataModule(hparams=args)
  model = MnistVAE(hparams=args)
  trainer = Trainer.from_argparse_args(args)
  trainer.fit(model, datamodule=dataloader)

  filename = 'mnist_vae.pt'
  path = f'{filename}'
  torch.save(model.state_dict(), path)
  # pprint(vars(args))