import time
import torch
import pytorch_lightning as pl
import options
from data_reader.dsprites import MultiDSpritesDataModule
from models.monet import MONet
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint


class LoggerCallback(Callback):

  def on_validation_epoch_end(self, trainer, pl_module):
    if pl_module.current_epoch == trainer.max_epochs - 1:
      codes = torch.cat(pl_module.codes)
      images = torch.cat(pl_module.images)
      pl_module.logger.experiment.add_embedding(
          codes, pl_module.labels, images, pl_module.current_epoch)
      pl_module.codes.clear()
      pl_module.labels.clear()
      pl_module.images.clear()


def train():
  pl.seed_everything(0)
  parser = options.get_parser()
  parser = pl.Trainer.add_argparse_args(parser)
  args = parser.parse_args()

  dataloader = MultiDSpritesDataModule(hparams=args)
  model = MONet(hparams=args)
  model.init_parameters()

  logger_callback = LoggerCallback()
  checkpoint_callback = ModelCheckpoint(
      save_top_k=1,
      monitor='train_loss',
      mode='min',
      prefix=model.__class__.__name__
  )
  early_stop_callback = EarlyStopping(
      monitor='train_loss',
      min_delta=0.01,
      patience=10,
      verbose=False,
      mode='min'
  )

  trainer = pl.Trainer.from_argparse_args(
      args,
      checkpoint_callback=checkpoint_callback,
      callbacks=[early_stop_callback],
      deterministic=True,
      log_every_n_steps=50,
      log_gpu_memory='min_max',
      # num_sanity_val_steps=0
  )

  start = time.time()
  trainer.fit(model, datamodule=dataloader)

  if not args.fast_dev_run:
    time_elapsed = time.time() - start
    h = time_elapsed // 3600
    m = (time_elapsed // 60) % 60
    s = time_elapsed % 60
    print(f'Training complete in {h:.0f}h {m:.0f}m {s:.0f}s')

  print(f'Lowest train error: {checkpoint_callback.best_model_score:.4f}')


if __name__ == '__main__':
  train()