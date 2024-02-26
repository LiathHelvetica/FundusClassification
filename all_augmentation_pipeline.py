import gc
from typing import Self

import cv2
import albumentations as al
from albumentations import BasicTransform, CenterCrop
from keras.src.engine.base_layer import Layer
from numpy import ndarray
from keras import layers
from pandas import read_csv
import itertools as it

from constants import ALL_LABEL_PATH, \
  ALL_DATA_PATH, AUGMENTATION_OUT_PATH
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
    elif callable(t):
      return self.apply_transform(t(self), t_name)
    else:
      raise Exception("Provided unsupported transform")
    return out

  def save_to(self, path: str):
    cv2.imwrite(f"{path}/{self.file_name}", cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR))


def get_crop_transform(fi: FundusImage) -> CenterCrop:
  x, y, _ = fi.image.shape
  min_dim = min(x, y)
  return al.CenterCrop(min_dim, min_dim)


def apply_transformation_to_batch(t, batch: list[FundusImage], t_name="") -> list[FundusImage]:
  return list(map(
    lambda img: img.apply_transform(t, t_name),
    batch
  ))


def center_pipeline(batch: list[FundusImage]) -> list[FundusImage]:
  batch = apply_transformation_to_batch(get_crop_transform, batch, t_name="")
  return batch


def basic_pipeline(batch: list[FundusImage]) -> list[FundusImage]:

  rotations = map(
    lambda f: FundusTransformation(layers.RandomRotation(
      f,
      fill_mode="constant"
    ), name=f"rot{int(f * 360)}"),
    [i * 0.0625 for i in range(0, 16)]
  )

  batch = flatten(map(lambda r: apply_transformation_to_batch(r.t, batch, r.name), rotations))

  batch = (batch +
           apply_transformation_to_batch(al.VerticalFlip(p=1), batch, t_name="flipx") +
           apply_transformation_to_batch(al.HorizontalFlip(p=1), batch, t_name="flipy"))

  return batch


def apply_shifts(batch: list[FundusImage]) -> list[FundusImage]:

  translations = map(
    lambda tuple: FundusTransformation(al.Affine(
      translate_percent={"x": tuple[0], "y": tuple[1]}, cval=0, p=1.0
    ), name=f"trx{int(tuple[0] * 100)}y{int(tuple[1] * 100)}"),
    it.product([-0.05, -0.025, 0.0, 0.025, 0.05], repeat=2)
  )

  batch = flatten(map(lambda r: apply_transformation_to_batch(r.t, batch, r.name), translations))

  return batch


# does not include original batch
def apply_zooms(batch: list[FundusImage]) -> list[FundusImage]:

  zooms = map(
    lambda z: FundusTransformation(al.Affine(
      scale=z, cval=0, p=1.0
    ), name=f"zoom{int(100 * z)}"),
    [0.9, 0.95, 1.05, 1.1]
  )

  batch = flatten(map(lambda r: apply_transformation_to_batch(r.t, batch, r.name), zooms))

  return batch


# does not include original batch
def apply_shears(batch: list[FundusImage]) -> list[FundusImage]:

  shears = map(
    lambda deg: FundusTransformation(al.Affine(
      shear=deg, cval=0, p=1.0
    ), name=f"shear{deg}"),
    [-20, -17, 17, 20]
  )

  batch = flatten(map(lambda r: apply_transformation_to_batch(r.t, batch, r.name), shears))

  return batch


def apply_distortions(batch: list[FundusImage]) -> list[FundusImage]:

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

  batch = flatten(map(lambda r: apply_transformation_to_batch(r.t, batch, r.name), g_distortions))

  return batch


def save_batch(batch: list[FundusImage], out_path: str) -> None:
  head = batch[0]
  print(f'saving batch from file: {head.file_name.split("-")[-1].split(".")[0]}')
  for im in batch:
    im.save_to(out_path)


OUT_SIZE = 224
ALL_OUT_PATH = AUGMENTATION_OUT_PATH
TARGET = 32000

if __name__ == "__main__":
  label_df = read_csv(ALL_LABEL_PATH, index_col="ID")
  disease_counts = label_df['Disease'].value_counts()
  for id, row in label_df.iterrows():
    disease = row["Disease"]
    disease_count = disease_counts[disease]
    f_name = f"{id}.png"
    data: ndarray = cv2.imread(f"{ALL_DATA_PATH}/{f_name}")
    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    img = FundusImage(data, f_name)
    batch = [img]
    print("basic pipeline")
    batch = center_pipeline(batch)
    batch = basic_pipeline(batch)
    if len(batch) >= TARGET / disease_count:
      save_batch(batch, ALL_OUT_PATH)
      continue

    print("shears")
    batch = batch + apply_shears(batch)
    if len(batch) >= TARGET / disease_count:
      save_batch(batch, ALL_OUT_PATH)
      continue

    print("zooms")
    batch = batch + apply_zooms(batch)
    if len(batch) >= TARGET / disease_count:
      save_batch(batch, ALL_OUT_PATH)
      continue

    # print("distortions")
    # batch = batch + apply_distortions(batch)
    # if len(batch) >= TARGET / disease_count:
    #   save_batch(batch, ALL_OUT_PATH)
    #   continue

    save_batch(batch, ALL_OUT_PATH)
    del batch
    gc.collect()
    # batch = apply_shifts(batch)




