import os
from math import inf
from typing import Self

import cv2
import albumentations as al
from albumentations import BasicTransform
from keras.src.engine.base_layer import Layer
from numpy import ndarray
from keras import layers

from constants import TEST_DATA_PATH, OUT_PATH
from utils import flatten


class FundusTransformation:
  def __init__(self, t: Layer | BasicTransform, name: str):
    self.t = t
    self.name = name


class FundusImage:

  def __init__(self, image: ndarray, file_name: str):
    self.image: ndarray = image
    self.file_name: str = file_name

  def apply_al_transform(self, t, t_name="") -> Self:
    out = FundusImage(self.image, f"{t_name}-{self.file_name}")
    out.image = t(image=out.image)["image"]
    return out

  def apply_kr_transform(self, t, t_name="") -> Self:
    out = FundusImage(self.image, f"{t_name}-{self.file_name}")
    out.image = t(out.image).numpy()
    return out

  def apply_transform(self, t: Layer | BasicTransform, t_name="") -> Self:
    out = FundusImage(self.image, f"{t_name}-{self.file_name}")
    if isinstance(t, Layer):
      out.image = t(out.image).numpy()
    elif isinstance(t, BasicTransform):
      out.image = t(image=out.image)["image"]
    else:
      raise Exception("Provided unsupported transform")
    return out

  def save_to(self, path: str):
    cv2.imwrite(f"{path}/{self.file_name}", cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR))


def read_imgs_from_path(path: str, batch_name="", n_limit=inf) -> list[FundusImage]:
  outcome = []
  files = os.listdir(path)
  n_files = min(float(len(files)), n_limit)
  for file in files[0:int(n_files)]:
    data: ndarray = cv2.imread(f"{path}/{file}")
    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    outcome.append(FundusImage(data, f"{batch_name}-{file}"))
  return outcome


def apply_transformation_to_batch(t, batch: list[FundusImage], t_name="") -> list[FundusImage]:
  return list(map(
    lambda img: img.apply_transform(t, t_name),
    batch
  ))


batch = read_imgs_from_path(TEST_DATA_PATH,  batch_name="test", n_limit=2)
rotations = map(
  lambda f: FundusTransformation(layers.RandomRotation(
    f,
    fill_mode="constant"
  ), name=f"rot{int(f * 360)}"),
  [i * 0.0625 for i in range(0, 16)]
)
batch = flatten(map(
  lambda rot: apply_transformation_to_batch(rot.t, batch, t_name=rot.name),
  rotations
))
batch = (batch +
         apply_transformation_to_batch(al.VerticalFlip(p=1), batch, t_name="flipx") +
         apply_transformation_to_batch(al.HorizontalFlip(p=1), batch, t_name="flipy"))

for im in batch:
  im.save_to(OUT_PATH)
print(len(batch))
