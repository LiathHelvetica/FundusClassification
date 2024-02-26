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


class FundusImage:

  def __init__(self, image: ndarray, file_name: str):
    self.image: ndarray = image
    self.file_name: str = file_name

  def save_to(self, path: str):
    cv2.imwrite(f"{path}/{self.file_name}", cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR))


# t - Layer (tensorflow) | BasicTransform (album) | f: ndarray -> ndarray
class FundusTransformation:

  def __init__(self, t, name: str):
    self.name = name
    if isinstance(t, Layer):
      self.f = lambda ndarr: t(ndarr).numpy()
    elif isinstance(t, BasicTransform):
      self.f = lambda ndarr: t(image=ndarr)["image"]
    elif callable(t):
      self.f = t
    else:
      raise Exception("Provided unsupported transform")

  def apply(self, fim: FundusImage) -> FundusImage:
    nd_out = self.f(fim.image)
    return FundusImage(nd_out, f"{self.name}-{fim.file_name}")

  def append(self, t, name=None) -> None:
    self.name = f"{name}-{self.name}"
    if isinstance(t, Layer):
      self.f = lambda ndarr: t(self.f(ndarr)).numpy()
    elif isinstance(t, BasicTransform):
      self.f = lambda ndarr: t(image=self.f(ndarr))["image"]
    elif callable(t):
      self.f = lambda ndarr: t(self.f(ndarr))
    elif isinstance(t, FundusTransformation):
      self.f = lambda ndarr: t.f(self.f(ndarr))
      self.name = f"{t.name}-{self.name}"
    else:
      raise Exception("Provided unsupported transform")

  def compose(self, t, name=None) -> Self:
    out_name = f"{name}-{self.name}"
    if isinstance(t, Layer):
      out_f = lambda ndarr: t(self.f(ndarr)).numpy()
    elif isinstance(t, BasicTransform):
      out_f = lambda ndarr: t(image=self.f(ndarr))["image"]
    elif callable(t):
      out_f = lambda ndarr: t(self.f(ndarr))
    elif isinstance(t, FundusTransformation):
      out_f = lambda ndarr: t.f(self.f(ndarr))
      out_name = f"{t.name}-{self.name}"
    else:
      raise Exception("Provided unsupported transform")
    return FundusTransformation(out_f, out_name)


def get_crop_transform(fi: ndarray) -> ndarray:
  x, y, _ = fi.shape
  min_dim = min(x, y)
  t = al.CenterCrop(min_dim, min_dim)
  return t(image=fi)["image"]


OUT_SIZE = 224
ALL_OUT_PATH = AUGMENTATION_OUT_PATH
TARGET = 32000

if __name__ == "__main__":

  rotations = map(
    lambda f: FundusTransformation(layers.RandomRotation(
      f,
      fill_mode="constant"
    ), name=f"rot{int(f * 360)}"),
    [i * 0.03125 for i in range(1, 32)]
  )

  transforms = [FundusTransformation(get_crop_transform, name="")]

  piv = []
  for rot, tr in it.product(rotations, transforms):
    piv.append(tr.compose(rot))
  transforms = transforms + piv


  label_df = read_csv(ALL_LABEL_PATH, index_col="ID")
  disease_counts = label_df['Disease'].value_counts()
  for id, row in label_df.iterrows():
    disease = row["Disease"]
    disease_count = disease_counts[disease]
    f_name = f"{id}.png"
    data: ndarray = cv2.imread(f"{ALL_DATA_PATH}/{f_name}")
    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    img = FundusImage(data, f_name)

    for tr in transforms:
      aug_img = tr.apply(img)
      aug_img.save_to(ALL_OUT_PATH)
