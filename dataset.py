import os
from pandas import read_csv
from torch.utils.data import Dataset
from torchvision.io import read_image

from utils import get_id_from_file_name


def get_column_name(row) -> str:
  for col in row.index:
    if row[col] == 1:
      return col
  return "OK"


class FundusImageDataset(Dataset):

  def __init__(self, img_path: str, label_path: str, exclude_labels: set[str] | None = None):
    if exclude_labels is None:
      exclude_labels = []
    self.img_path = img_path
    self.label_path = label_path
    self.label_df = read_csv(label_path)
    self.label_df["Disease"] = self.label_df.drop(["ID", "Disease_Risk"], axis=1).apply(get_column_name, axis=1)
    self.label_df = self.label_df[["ID", "Disease"]]
    self.label_df = self.label_df[self.label_df.apply(lambda r: r["Disease"] not in exclude_labels, axis=1)]
    self.images = []
    for f_name in os.listdir(img_path):
      id = get_id_from_file_name(f_name)
      if id in self.label_df["ID"].values:
        self.images.append(f_name)

  def __len__(self):
    len(self.images)

  def __getitem__(self, index):
    f_name = self.images[index]
    id = get_id_from_file_name(f_name)
    return read_image(f"{self.img_path}/{f_name}"), self.label_df[self.label_df["ID"] == id].iloc[0]["Disease"]
