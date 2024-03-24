import random
from copy import copy
from typing import Self

import cv2
import albumentations as al
from albumentations import BasicTransform, CenterCrop
from numpy import ndarray
from keras import layers, Layer
from pandas import read_csv
import itertools as it
import tensorflow as tf

from constants import ALL_LABEL_PATH, \
  ALL_DATA_PATH, AUGMENTATION_OUT_PATH, OUT_PATH, ALL_CROPPED_SQUARE_DATA_PATH
from utils import try_or_else


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


def identity_transform(fi: ndarray) -> ndarray:
  return fi


def get_to_square_transform(fi: ndarray) -> ndarray:
  size = max(fi.shape[0], fi.shape[1])
  out = tf.image.resize_with_pad(fi, size, size).numpy()
  return out


def get_rotation_transform(deg: int):
  def out(fi: ndarray) -> ndarray:
    (h, w) = fi.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), deg, 1.0)
    return cv2.warpAffine(fi, M, (w, h))
  return out



OUT_SIZE = 224
ALL_OUT_TRAIN_PATH = f"{OUT_PATH}/morealltrain{OUT_SIZE}"
ALL_OUT_VAL_PATH = f"{OUT_PATH}/moreallval{OUT_SIZE}"
TRAIN_TARGET = 17600
VAL_TARGET = 4400

if __name__ == "__main__":

  resize_transform = FundusTransformation(
    get_to_square_transform,
    name=""
  ).compose(
    al.Resize(OUT_SIZE, OUT_SIZE, p=1.0),
    name=f"res{OUT_SIZE}"
  )

  rotations = map(
    lambda f: FundusTransformation(get_rotation_transform(f), name=f"rot{f}"),
    [int(i * 0.03125 * 360) for i in range(1, 32)]
  )

  shears = [
    FundusTransformation(al.CropAndPad(percent=(0.02, 0.0, 0.02, 0.0)), name="shear(18)").compose(
      al.Affine(
        shear=18, cval=0, p=1.0
      ), name=""
    ),
    FundusTransformation(al.CropAndPad(percent=(0.05, 0.0, 0.05, 0.0)), name="shear(25)").compose(
      al.Affine(
        shear=25, cval=0, p=1.0
      ), name=""
    ),
    FundusTransformation(al.CropAndPad(percent=(0.12, 0.0, 0.12, 0.0)), name="shear(32)").compose(
      al.Affine(
        shear=32, cval=0, p=1.0
      ), name=""
    ),
    FundusTransformation(al.CropAndPad(percent=(0.02, 0.0, 0.02, 0.0)), name="shear(-18)").compose(
      al.Affine(
        shear=-18, cval=0, p=1.0
      ), name=""
    ),
    FundusTransformation(al.CropAndPad(percent=(0.05, 0.0, 0.05, 0.0)), name="shear(-25)").compose(
      al.Affine(
        shear=-25, cval=0, p=1.0
      ), name=""
    ),
    FundusTransformation(al.CropAndPad(percent=(0.12, 0.0, 0.12, 0.0)), name="shear(-32)").compose(
      al.Affine(
        shear=-32, cval=0, p=1.0
      ), name=""
    )
  ]
  
  safe_zooms = map(
    lambda z: FundusTransformation(al.Affine(
      scale=z, cval=0, p=1.0
    ), name=f"zoom{int(100 * z)}"),
    [0.9, 0.95]
  )

  unsafe_zooms = map(
    lambda z: FundusTransformation(al.Affine(
      scale=z, cval=0, p=1.0
    ), name=f"zoom{int(100 * z)}"),
    [1.05, 1.10]
  )

  translations = list(filter(lambda tr: tr is not None, map(
    lambda tuple: FundusTransformation(al.Affine(
      translate_percent={"x": tuple[0], "y": tuple[1]}, cval=0, p=1.0
    ), name=f"trx{int(tuple[0] * 1000)}y{int(tuple[1] * 1000)}") if tuple != (0.0, 0.0) else None,
    it.product([-0.045,  0.0, 0.045], repeat=2)
  )))

  px_dropouts = map(
    lambda p: FundusTransformation(al.PixelDropout(
      dropout_prob=p, p=1.0
    ), name=f"pxdrop{int(1000 * p)}"),
    [0.005, 0.0075, 0.01] # 0.025
  )

  crs_dropouts = map(
    lambda tpl: FundusTransformation(al.CoarseDropout(
      max_holes=tpl[0], max_height=tpl[1], max_width=tpl[1], p=1.0
    ), name=f"crsdrp({tpl[0]},{tpl[1]})"),
    [
      (30, 2),
      (40, 2),
      (50, 2),
      (10, 4),
      (20, 4),
      (30, 4),
      (10, 6),
      (20, 6),
    ]
  ) 

  mixed_dropouts = [
    FundusTransformation(al.CoarseDropout(max_holes=10, max_height=4, max_width=4, p=1.0), name="mixdrp1").compose(
      al.PixelDropout(dropout_prob=0.005, p=1.0), name=""
    ),
    FundusTransformation(al.CoarseDropout(max_holes=10, max_height=4, max_width=4, p=1.0), name="mixdrp1").compose(
      al.PixelDropout(dropout_prob=0.0075, p=1.0), name=""
    ),
    FundusTransformation(al.CoarseDropout(max_holes=20, max_height=4, max_width=4, p=1.0), name="mixdrp1").compose(
      al.PixelDropout(dropout_prob=0.005, p=1.0), name=""
    ),
    FundusTransformation(al.CoarseDropout(max_holes=20, max_height=4, max_width=4, p=1.0), name="mixdrp1").compose(
      al.PixelDropout(dropout_prob=0.0075, p=1.0), name=""
    ),
    FundusTransformation(al.CoarseDropout(max_holes=30, max_height=4, max_width=4, p=1.0), name="mixdrp1").compose(
      al.PixelDropout(dropout_prob=0.005, p=1.0), name=""
    ),
    FundusTransformation(al.CoarseDropout(max_holes=30, max_height=4, max_width=4, p=1.0), name="mixdrp1").compose(
      al.PixelDropout(dropout_prob=0.0075, p=1.0), name=""
    ),
  ]

  transforms = [FundusTransformation(identity_transform, name="id")]

  piv = []
  for rot, tr in it.product(rotations, transforms):
    piv.append(tr.compose(rot))
  transforms = transforms + piv

  piv = []
  for tr in transforms:
    piv.append(tr.compose(al.VerticalFlip(p=1), name="xflip"))
    piv.append(tr.compose(al.HorizontalFlip(p=1), name="yflip"))
  transforms = transforms + piv

  piv = []
  for shear, tr in it.product(shears, transforms):
    piv.append(tr.compose(shear))
  transforms = transforms + piv

  zooms_piv = []
  for zoom, tr in it.product(safe_zooms, transforms):
    zooms_piv.append(tr.compose(zoom))
  transforms = transforms + zooms_piv

  safe_transforms_1 = transforms.copy()

  trans_piv = []
  for trans, tr in it.product(translations, transforms):
    trans_piv.append(tr.compose(trans))
  transforms = transforms + trans_piv

  zooms_piv = []
  for zoom, tr in it.product(unsafe_zooms, safe_transforms_1):
      zooms_piv.append(tr.compose(zoom))
  transforms = transforms + zooms_piv

  labels_df = read_csv(ALL_LABEL_PATH)

  # disease -> list[id]
  id_dict: dict[str, list[str]] = {}
  for id, row in labels_df.iterrows():
    ids_for_disease = id_dict.get(row["Disease"])
    if ids_for_disease is None:
      id_dict[row["Disease"]] = [row["ID"]]
    else:
      ids_for_disease.append(row["ID"])

  # disease -> which ids are in which set (val, train)
  id_split: dict[str, dict[str, set[str]]] = {}
  for disease, id_list in id_dict.items():
    id_list_shuff = copy(id_list)
    random.shuffle(id_list_shuff)
    val_size = int(0.15 * len(id_list_shuff))
    train_size = len(id_list_shuff) - val_size
    train_ids = id_list_shuff[0:train_size]
    val_ids = id_list_shuff[train_size:]
    if len(val_ids) == 0 and len(train_ids) == 1:
      val_ids = [train_ids[0]]
    elif len(val_ids) == 0:
      val_ids = [train_ids[0]]
      train_ids = train_ids[1:]
    id_split[disease] = {
      "train": set(train_ids),
      "val": set(val_ids)
    }

  for disease, set_split in id_split.items():

    train_set: set[str] = set_split["train"]
    val_set: set[str] = set_split["val"]
    both_set = train_set.intersection(val_set)
    train_set = train_set - both_set
    val_set = val_set - both_set

    train_target_per_id = try_or_else(lambda: int(TRAIN_TARGET / len(train_set)) + 1, TRAIN_TARGET)
    val_target_per_id = try_or_else(lambda: int(VAL_TARGET / len(val_set)) + 1, VAL_TARGET)

    for id in train_set:
      f_name = f"{id}.png"
      data: ndarray = cv2.imread(f"{ALL_CROPPED_SQUARE_DATA_PATH}/{f_name}")
      data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
      img = FundusImage(data, f_name)
      t_selected = random.sample(transforms, train_target_per_id)

      for tr in t_selected:
        tr = tr.compose(resize_transform, name=f"res{OUT_SIZE}")
        aug_img = tr.apply(img)
        aug_img.save_to(ALL_OUT_TRAIN_PATH)
      print(f"> {id}")


    for id in val_set:
      f_name = f"{id}.png"
      data: ndarray = cv2.imread(f"{ALL_CROPPED_SQUARE_DATA_PATH}/{f_name}")
      data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
      img = FundusImage(data, f_name)
      t_selected = random.sample(transforms, val_target_per_id)

      for tr in t_selected:
        tr = tr.compose(resize_transform, name=f"res{OUT_SIZE}")
        aug_img = tr.apply(img)
        aug_img.save_to(ALL_OUT_VAL_PATH)
      print(f"> {id}")

    for id in both_set:
      f_name = f"{id}.png"
      data: ndarray = cv2.imread(f"{ALL_CROPPED_SQUARE_DATA_PATH}/{f_name}")
      data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
      img = FundusImage(data, f_name)
      t_selected = random.sample(transforms, train_target_per_id + val_target_per_id)
      t_val_selected = t_selected[0:val_target_per_id]
      t_train_selected = t_selected[val_target_per_id:]

      for tr in t_train_selected:
        tr = tr.compose(resize_transform, name=f"res{OUT_SIZE}")
        aug_img = tr.apply(img)
        aug_img.save_to(ALL_OUT_TRAIN_PATH)

      for tr in t_val_selected:
        tr = tr.compose(resize_transform, name=f"res{OUT_SIZE}")
        aug_img = tr.apply(img)
        aug_img.save_to(ALL_OUT_VAL_PATH)
      print(f"> {id}")

