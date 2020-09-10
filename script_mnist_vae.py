
def main():
  import time
  import torch
  import pytorch_lightning as pl
  from argparse import ArgumentParser
  from pytorch_lightning.callbacks import ModelCheckpoint

  from models.mnist_vae import MnistVAE
  from data_reader.mnist import MnistDataModule


  # torch.manual_seed(0)
  # torch.cuda.manual_seed(0)
  pl.seed_everything(0)

  parser = ArgumentParser()
  parser = pl.Trainer.add_argparse_args(parser)
  # parser.add_argument('--model_path', 
  #     default='/content/drive/My Drive/models/mnist_vae.pt', type=str)
  parser.add_argument('--output_dir', default='./logs', type=str)
  parser.add_argument('--dataset_dir', default='./datasets', type=str)
  parser.add_argument('--num_workers', default=4, type=int)

  parser.add_argument('--batch_size', default=64, type=int)
  parser.add_argument('--learning_rate', default=1e-3, type=float)

  parser.add_argument('--input_channels', default=1, type=int)
  parser.add_argument('--input_height', default=28, type=int)
  parser.add_argument('--input_width', default=28, type=int)
  parser.add_argument('--output_channels', default=1, type=int)
  parser.add_argument('--beta', default=0.5, type=float)
  parser.add_argument('--z_dim', default=16, type=int)

  args = parser.parse_args()

  dataloader = MnistDataModule(hparams=args)
  model = MnistVAE(hparams=args)

  checkpoint_callback = ModelCheckpoint(
      save_top_k=1,
      monitor='val_loss',
      mode='min',
      prefix=model.__class__.__name__+'_'
  )

  trainer = pl.Trainer.from_argparse_args(
      args,
      checkpoint_callback=checkpoint_callback,
      progress_bar_refresh_rate=20
  )

  since = time.time()
  trainer.fit(model, datamodule=dataloader)

  if not args.fast_dev_run:
    time_elapsed = time.time() - since
    h = time_elapsed // 3600
    m = (time_elapsed // 60) % 60
    s = time_elapsed % 60
    print(f'Training complete in {h:.0f}h {m:.0f}m {s:.0f}s')
    # torch.save(model.state_dict(), args.model_path)

  print(f'Best model score: {checkpoint_callback.best_model_score:.4f}')


if __name__ == '__main__':
  main()