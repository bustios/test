from argparse import ArgumentParser


def get_parser():
  parser = ArgumentParser()
  parser.add_argument('--dataset_dir', default='./datasets', type=str)
  parser.add_argument('--num_workers', default=4, type=int)

  parser.add_argument('--batch_size', default=32, type=int)
  parser.add_argument('--learning_rate', default=1e-3, type=float)

  parser.add_argument('--image_type', default=0, type=int)
  parser.add_argument('--input_channels', default=1, type=int)
  parser.add_argument('--input_height', default=28, type=int)
  parser.add_argument('--input_width', default=28, type=int)
  parser.add_argument('--output_channels', default=1, type=int)
  parser.add_argument('--beta', default=0.5, type=float)
  parser.add_argument('--z_dim', default=16, type=int)
  return parser