
if __name__ == '__main__':
  import time
  import torch
  from argparse import ArgumentParser
  from pytorch_lightning import Trainer

  from models.mnist_vae import MnistVAE
  from data_reader.mnist import MnistDataModule


  parser = ArgumentParser()
  parser = Trainer.add_argparse_args(parser)
  parser.add_argument('--model_path', 
      default='/content/drive/My Drive/experiments/mnist_vae.pt', type=str)
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

  args = parser.parse_args()

  dataloader = MnistDataModule(hparams=args)
  model = MnistVAE(hparams=args)
  trainer = Trainer.from_argparse_args(args)

  since = time.time()
  trainer.fit(model, datamodule=dataloader)
  time_elapsed = time.time() - since
  h = time_elapsed // 3600
  m = (time_elapsed // 60) % 60
  s = time_elapsed % 60
  print(f'Training complete in {h:.0f}h {m:.0f}m {s:.0f}s')
  
  torch.save(model.state_dict(), args.model_path)