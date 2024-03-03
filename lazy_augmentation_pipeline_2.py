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
  ALL_DATA_PATH, AUGMENTATION_OUT_PATH, OUT_PATH


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


OUT_SIZE = 224
ALL_OUT_PATH = f"{OUT_PATH}/all{OUT_SIZE}"
TARGET = 32000

if __name__ == "__main__":

  rotations = map(
    lambda f: FundusTransformation(layers.RandomRotation(
      f,
      fill_mode="constant"
    ), name=f"rot{int(f * 360)}"),
    [i * 0.03125 for i in range(1, 32)]
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
    [0.105, 0.110]
  )

  translations = list(filter(lambda tr: tr is not None, map(
    lambda tuple: FundusTransformation(al.Affine(
      translate_percent=(tuple[0], tuple[1]), cval=0, p=1.0
    ), name=f"trx{int(tuple[0] * 1000)}y{int(tuple[1] * 1000)}") if tuple != (0.0, 0.0) else None,
    it.product([-0.045, -0.025,  0.0, 0.025, 0.045], repeat=2)
  ))) + list(filter(lambda tr: tr is not None, map(
    lambda v: FundusTransformation(al.Affine(
      translate_percent=(v, 0.045), cval=0, p=1.0
    ), name=f"trx{int(v * 1000)}y{45}") if v != (0.0, 0.0) else None,
    [-0.025, 0.0, 0.025]
  ))) + list(filter(lambda tr: tr is not None, map(
    lambda v: FundusTransformation(al.Affine(
      translate_percent=(0.045, v), cval=0, p=1.0
    ), name=f"trx{45}y{int(v * 1000)}") if v != (0.0, 0.0) else None,
    [-0.025, 0.0, 0.025]
  )))

  resize_transform = FundusTransformation(al.Resize(OUT_SIZE, OUT_SIZE, p=1.0), name=f"res{OUT_SIZE}")

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

  transforms = [FundusTransformation(get_crop_transform, name="c")]

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
  safe_transforms_1 = transforms.copy()

  trans_piv = []
  for trans, tr in it.product(translations, transforms):
    trans_piv.append(tr.compose(trans))
  transforms = transforms + trans_piv

  zooms_piv = []
  for zoom, tr in it.product(safe_zooms, transforms):
    zooms_piv.append(tr.compose(zoom))
  transforms = transforms + zooms_piv

  zooms_piv = []
  for zoom, tr in it.product(unsafe_zooms, safe_transforms_1):
    zooms_piv.append(tr.compose(zoom))
  transforms = transforms + zooms_piv

  transforms = list(map(lambda tr: tr.compose(resize_transform), transforms))

  piv_px_dr = []
  for pxdr, tr in it.product(px_dropouts, transforms):
    piv_px_dr.append(tr.compose(pxdr))

  piv_crs_dr = []
  for crsdrp, tr in it.product(crs_dropouts, transforms):
    piv_crs_dr.append(tr.compose(crsdrp))

  transforms = transforms + piv_crs_dr + piv_px_dr

  label_df = read_csv(ALL_LABEL_PATH, index_col="ID")
  disease_counts = label_df['Disease'].value_counts()
  for id, row in label_df.iterrows():

    disease = row["Disease"]
    disease_count = disease_counts[disease]
    it_target = int(TARGET / disease_count) + 1

    f_name = f"{id}.png"
    data: ndarray = cv2.imread(f"{ALL_DATA_PATH}/{f_name}")
    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    img = FundusImage(data, f_name)

    for tr in transforms[0:it_target]:
      tr = tr.compose(resize_transform, name=f"res{OUT_SIZE}")
      aug_img = tr.apply(img)
      aug_img.save_to(ALL_OUT_PATH)
