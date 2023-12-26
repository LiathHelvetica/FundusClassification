import os
import datetime
from math import inf
from typing import Self

import cv2
import albumentations as al
import itertools as based
from albumentations import BasicTransform, CenterCrop
from keras.src.engine.base_layer import Layer
from numpy import ndarray
from keras import layers

from constants import TEST_DATA_PATH, OUT_PATH


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
    elif callable(t):
      return self.apply_transform(t(self), t_name)
    else:
      raise Exception("Provided unsupported transform")
    return out

  def save_to(self, path: str):
    cv2.imwrite(f"{path}/{self.file_name}", cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR))


class FundusBatch:

  def __init__(self, base_path: str, out_path: str, namespace=None):
    if namespace is None:
      namespace = []
    self.namespace = namespace
    self.base_path = base_path
    self.out_path = out_path

  def __iter__(self):
    self.i = 0
    return self

  def __next__(self) -> FundusImage:
    if self.i < len(self.namespace):
      f_name = self.namespace[self.i]
      path = f"{self.base_path}/{f_name}"
      data: ndarray = cv2.imread(path)
      data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
      self.i = self.i + 1
      return FundusImage(data, f_name)
    else:
      raise StopIteration

  def __add__(self, other):
    assert isinstance(other, type(self))
    assert self.base_path == other.base_path
    assert self.out_path == other.out_path
    return FundusBatch(self.base_path, self.out_path, self.namespace + other.namespace)

  def copy(self):
    return FundusBatch(self.base_path, self.out_path, self.namespace.copy())


"""
def read_imgs_from_path(path: str, batch_name="", n_limit=inf) -> list[FundusImage]:
  outcome = []
  files = os.listdir(path)
  n_files = min(float(len(files)), n_limit)
  for file in files[0:int(n_files)]:
    data: ndarray = cv2.imread(f"{path}/{file}")
    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    outcome.append(FundusImage(data, f"{batch_name}-{file}"))
  return outcome
"""


def read_batch_from_path(path: str, n_limit=inf) -> FundusBatch:
  outcome = []
  files = os.listdir(path)
  n_files = min(float(len(files)), n_limit)
  for file in files[0:int(n_files)]:
    outcome.append(f"{file}")
  return FundusBatch(path, OUT_PATH, outcome)


def apply_transformation_to_batch(t, batch: FundusBatch, t_name="") -> FundusBatch:
  f_names_out = []
  for fi in batch:
    fi_out = fi.apply_transform(t, t_name)
    fi_out.save_to(batch.out_path)
    f_names_out.append(fi_out.file_name)
  return FundusBatch(batch.base_path, batch.out_path, f_names_out)


def add_transformation_to_batch(t, batch: FundusBatch, t_name="") -> FundusBatch:
  f_names_out = []
  for fi in batch:
    fi_out = fi.apply_transform(t, t_name)
    fi_out.save_to(batch.out_path)
    f_names_out.append(fi_out.file_name)
  return FundusBatch(batch.base_path, batch.out_path, batch.namespace + f_names_out)


def get_crop_transform(fi: FundusImage) -> CenterCrop:
  x, y, _ = fi.image.shape
  min_dim = min(x, y)
  return al.CenterCrop(min_dim, min_dim)

########################################################################################################################


start = datetime.datetime.now()

batch = read_batch_from_path(TEST_DATA_PATH, n_limit=2)
batch = apply_transformation_to_batch(get_crop_transform, batch)
batch.base_path = batch.out_path


rotations = map(
  lambda f: FundusTransformation(layers.RandomRotation(
    f,
    fill_mode="constant"
  ), name=f"rot{int(f * 360)}"),
  [i * 0.0625 for i in range(1, 16)]
)

batch2 = batch.copy()
for rot in rotations:
  batch2 = batch2 + apply_transformation_to_batch(rot.t, batch, t_name=rot.name)


"""
translations = map(
  lambda tuple: FundusTransformation(al.Affine(
    translate_percent=(tuple[0], tuple[1]), cval=0, p=1.0
  ), name=f"trx{int(tuple[0] * 100)}y{int(tuple[1] * 100)}"),
  based.product([-0.05, -0.025, 0.025, 0.05], repeat=2)
)

batch3 = batch2.copy()
for tran in translations:
  batch3 = batch3 + apply_transformation_to_batch(tran.t, batch2, t_name=tran.name)
"""

zooms = map(
  lambda z: FundusTransformation(al.Affine(
    scale=z, cval=0, p=1.0
  ), name=f"zoom{int(100 * z)}"),
  [0.9, 0.95, 1.05, 1.1]
)

batch3 = batch2
batch4 = batch3.copy()
for zoom in zooms:
  batch4 = batch4 + apply_transformation_to_batch(zoom.t, batch3, t_name=zoom.name)

batch5 = apply_transformation_to_batch(al.VerticalFlip(p=1), batch4, t_name="flipx")
batch6 = apply_transformation_to_batch(al.HorizontalFlip(p=1), batch4, t_name="flipy")

batch7 = batch5 + batch6 + batch4


"""
shears = map(
  lambda deg: FundusTransformation(al.Affine(
    shear=deg, cval=0, p=1.0
  ), name=f"shear{deg}"),
  [-20, -15, 15, 20]
)

batch8 = batch7.copy()
for shear in shears:
  batch8 = batch8 + apply_transformation_to_batch(shear.t, batch7, t_name=shear.name)
"""


dropouts = map(
  lambda tpl: FundusTransformation(al.CoarseDropout(
    max_holes=tpl[0], min_holes=tpl[0], max_width=tpl[1], min_width=tpl[1], max_height=tpl[1], min_height=tpl[1], fill_value=0, p=1.0
  ), name=f"dropout(n={tpl[0]},l={int(100 * tpl[1])})"),
  [
    (60, 0.01),
    (100, 0.01),
    (200, 0.005),
    (400, 0.005)
  ]
)

batch8 = batch7
batch9 = batch8.copy()
for dropout in dropouts:
  batch9 = batch9 + apply_transformation_to_batch(dropout.t, batch8, t_name=dropout.name)

""""
g_distortions = map(
  lambda tpl: FundusTransformation(al.GridDistortion(
    num_steps=tpl[0], distort_limit=tpl[1], p=1.0, normalized=True
  ), name=f"gdist(n={tpl[0]}dis={int(100 * tpl[1])})"),
  [
    (5, 0.3),
    (6, 0.3),
    (7, 0.3),
    (10, 0.5),
    (9, 0.5),
    (7, 0.5)
  ]
)

batch10 = batch9.copy()
for dist in g_distortions:
  batch10 = batch10 + apply_transformation_to_batch(dist.t, batch9, dist.name)
"""

px_dropouts = map(
  lambda prob: FundusTransformation(al.PixelDropout(
    dropout_prob=prob, p=1.0
  ), name=f"pxdrop{int(100 * prob)}"),
  [0.01, 0.025, 0.05]
)

batch10 = batch9
batch11 = batch10.copy()
for px_d in px_dropouts:
  batch11 = batch11 + apply_transformation_to_batch(px_d.t, batch10, px_d.name)





"""
elastics = map(
  lambda tpl: FundusTransformation(al.ElasticTransform(
    alpha=tpl[0], sigma=tpl[1], alpha_affine=tpl[2], interpolation=tpl[3], border_mode=tpl[4], p=1.0
  ), name=f"elas({tpl[0]}{tpl[1]}{tpl[2]}{tpl[3]}{tpl[4]})"),
  [
    (10, 10, 20, cv2.INTER_LANCZOS4, cv2.BORDER_REFLECT_101)
  ]
)

batch10 = batch9.copy()
for el in elastics:
  batch10 = batch10 + apply_transformation_to_batch(el.t, batch9, el.name)
"""


end = datetime.datetime.now()
print(f"Time: {end - start}")
