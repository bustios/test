from argparse import ArgumentParser


def get_parser():
  parser = ArgumentParser()
  parser.add_argument('--dataset_dir', default='./datasets', type=str)
  parser.add_argument('--dataset_size', default=0.1, type=float)
  parser.add_argument('--num_workers', default=4, type=int)

  parser.add_argument('--batch_size', default=64, type=int)
  parser.add_argument('--learning_rate', default=1e-4, type=float)

  parser.add_argument('--input_channels', default=3, type=int)
  parser.add_argument('--input_height', default=64, type=int)
  parser.add_argument('--input_width', default=64, type=int)
  parser.add_argument('--attention_out_channels', default=1, type=int)
  parser.add_argument('--z_dim', default=16, type=int)
  parser.add_argument('--beta', default=0.5, type=float)
  parser.add_argument('--gamma', default=0.5, type=float)
  parser.add_argument('--background_std', default=0.09, type=float)
  parser.add_argument('--foreground_std', default=0.11, type=float)

  parser.add_argument('--num_slots', default=5, type=int)
  return parser