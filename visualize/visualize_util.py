# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for the visualization code."""
import imageio
import numpy as np
import scipy.stats
import torch
import torchvision

@torch.no_grad()
def save_animation(list_of_animated_images, image_path, fps):
  full_size_images = []
  for single_images in zip(*list_of_animated_images):
    image_grid = torchvision.utils.make_grid(list(single_images), pad_value=1)
    image_grid = image_grid.permute(1, 2, 0).numpy()
    image_grid *= 255.
    image_grid = image_grid.astype("uint8")
    full_size_images.append(image_grid)
  imageio.mimwrite(image_path, full_size_images, fps=fps)


def cycle_factor(starting_index, num_indices, num_frames):
  """Cycles through the state space in a single cycle."""
  grid = np.linspace(starting_index, starting_index + 2*num_indices,
                     num=num_frames, endpoint=False)
  grid = np.array(np.ceil(grid), dtype=np.int64)
  grid -= np.maximum(0, 2*grid - 2*num_indices + 1)
  grid += np.maximum(0, -2*grid - 1)
  return grid


def cycle_gaussian(starting_value, num_frames, loc=0., scale=1.):
  """Cycles through the quantiles of a Gaussian in a single cycle."""
  starting_prob = scipy.stats.norm.cdf(starting_value, loc=loc, scale=scale)
  grid = np.linspace(starting_prob, starting_prob + 2.,
                     num=num_frames, endpoint=False)
  grid -= np.maximum(0, 2*grid - 2)
  grid += np.maximum(0, -2*grid)
  grid = np.minimum(grid, 0.999)
  grid = np.maximum(grid, 0.001)
  return np.array([scipy.stats.norm.ppf(i, loc=loc, scale=scale) for i in grid])


def cycle_interval(starting_value, num_frames, min_val, max_val):
  """Cycles through the state space in a single cycle."""
  starting_in_01 = (starting_value - min_val)/(max_val - min_val)
  grid = np.linspace(starting_in_01, starting_in_01 + 2.,
                     num=num_frames, endpoint=False)
  grid -= np.maximum(0, 2*grid - 2)
  grid += np.maximum(0, -2*grid)
  return grid * (max_val - min_val) + min_val