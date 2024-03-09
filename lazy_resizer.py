import cv2
import albumentations as al
from numpy import ndarray
from pandas import read_csv

from constants import ALL_LABEL_PATH, ALL_DATA_PATH, OUT_PATH
from lazy_augmentation_pipeline_2 import FundusTransformation, get_crop_transform, FundusImage

OUT_SIZE = 224
OUT_RESIZE_PATH = f"{OUT_PATH}/res{OUT_SIZE}"

if __name__ == "__main__":

  transform = (FundusTransformation(get_crop_transform, name="c")
    .compose(al.Resize(OUT_SIZE, OUT_SIZE, p=1.0), name=f"res{OUT_SIZE}"))

  label_df = read_csv(ALL_LABEL_PATH, index_col="ID")
  for id, row in label_df.iterrows():

    f_name = f"{id}.png"
    data: ndarray = cv2.imread(f"{ALL_DATA_PATH}/{f_name}")
    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    img = FundusImage(data, f_name)
    aug_img = transform.apply(img)
    aug_img.save_to(OUT_RESIZE_PATH)
