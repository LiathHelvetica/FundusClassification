import json
import os
from pandas import read_csv
from torch.utils.data import Dataset
from torchvision.io import read_image
from torch import tensor, Tensor
import torch
import torchvision.transforms as transforms

from utils import get_id_from_file_name


def get_column_name(row) -> str:
  for col in row.index:
    if row[col] == 1:
      return col
  return "OK"


class FundusImageDataset(Dataset):

  def __init__(
    self,
    img_path: str,
    label_path: str,
    exclude_labels: set[str] | None = None,
    x_size: int = 224,
    y_size: int = 224,
    force_same_class_representation: bool = True
  ):
    if exclude_labels is None:
      exclude_labels = []
    self.x_size = x_size
    self.y_size = y_size
    self.img_path = img_path
    self.label_path = label_path
    self.label_df = read_csv(label_path)
    self.label_df["Disease"] = self.label_df.drop(["ID", "Disease_Risk"], axis=1).apply(get_column_name, axis=1)
    self.label_df = self.label_df[["ID", "Disease"]]
    self.label_df = self.label_df[self.label_df.apply(lambda r: r["Disease"] not in exclude_labels, axis=1)]
    self.local_labels = []
    self.label_dict = dict()
    self.set_labels(list(self.label_df["Disease"].unique()))
    self.images = []
    for f_name in os.listdir(img_path):
      id = get_id_from_file_name(f_name)
      if id in self.label_df["ID"].values:
        self.images.append(f_name)
    if force_same_class_representation:
      max_n_repr = self.get_max_n_repr()
      acc: dict[str, int] = dict()
      override_out = []
      for f_name in self.images:
        id = get_id_from_file_name(f_name)
        disease = self.get_label_by_id(id)
        count = acc.get(disease)
        if count is None:
          override_out.append(f_name)
          acc[disease] = 1
        elif count <= max_n_repr:
          override_out.append(f_name)
          acc[disease] = acc[disease] + 1
        else:
          pass
      self.images = override_out

  def set_labels(self, labels: list[str]):
    self.local_labels = labels
    self.label_dict = {str(item): idx for idx, item in enumerate(self.local_labels)}

  def get_max_n_repr(self) -> int:
    acc: dict[str, int] = dict()
    for f_name in self.images:
      id = get_id_from_file_name(f_name)
      disease = self.get_label_by_id(id)
      if acc.get(disease) is not None:
        acc[disease] = acc[disease] + 1
      else:
        acc[disease] = 1
    counts = list(acc.values())
    return 0 if len(counts) == 0 else max(counts)

  def get_label_by_id(self, id) -> str:
    return self.label_df[self.label_df["ID"] == id].iloc[0]["Disease"]

  def get_label_int_by_id(self, id) -> int:
    return self.label_dict[self.get_label_by_id(id)]

  def __len__(self) -> int:
    return len(self.images)

  def __getitem__(self, index):
    f_name = self.images[index]
    id = get_id_from_file_name(f_name)
    data = read_image(f"{self.img_path}/{f_name}")
    resize_transform = transforms.Compose([
      transforms.ToPILImage(),
      transforms.Resize((self.x_size, self.y_size)),
      transforms.ToTensor()
    ])
    return resize_transform(data), self.get_label_int_by_id(id)
