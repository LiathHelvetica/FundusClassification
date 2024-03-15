from pandas import read_csv
from torch.utils.data import Dataset
from torchvision.io import read_image
from torch import Tensor
import torchvision.transforms as transforms

from constants import HEALTHY_LABEL
from utils import get_id_from_f_name, get_class_for_id


class FundusImageDataset2(Dataset):

  def __init__(
    self,
    img_path: str,
    img_list: list[str],
    label_path: str
  ):
    self.img_path = img_path
    self.img_list = img_list
    self.label_path = label_path
    self.label_df = read_csv(label_path)
    self.label_dict = {item: idx for idx, item in enumerate(list(self.label_df["Disease"].unique()))}


  def get_full_path(self, f_name: str) -> str:
    return f"{self.img_path}/{f_name}"

  def count_class_representation(self) -> dict[str, int]:
    acc: dict[str, int] = dict()
    for f_name in self.img_list:
      id = get_id_from_f_name(f_name)
      disease = self.get_label_by_id(id)
      count = acc.get(disease)
      if count is None:
        acc[disease] = 1
      else:
        acc[disease] = acc[disease] + 1
    return acc

  def get_label_by_id(self, id: str) -> str:
    return get_class_for_id(id, self.label_df)

  def get_label_int_by_id(self, id: str) -> int:
    return self.label_dict[self.get_label_by_id(id)]

  def get_ok_label_id(self) -> int:
    return self.label_dict[HEALTHY_LABEL]

  def get_all_labels(self) -> list[str]:
    def iter(f_name):
      id = get_id_from_f_name(f_name)
      return self.get_label_by_id(id)
    return list(map(lambda f_name: iter(f_name), self.img_list))

  def get_all_int_labels(self) -> list[int]:
    def iter(f_name):
      id = get_id_from_f_name(f_name)
      return self.get_label_int_by_id(id)
    return list(map(lambda f_name: iter(f_name), self.img_list))

  def __len__(self) -> int:
    return len(self.img_list)

  def __getitem__(self, index) -> (Tensor, int):
    f_name = self.img_list[index]
    id = get_id_from_f_name(f_name)
    data = read_image(f"{self.img_path}/{f_name}")
    transform = transforms.Compose([
      transforms.ToPILImage(),
      transforms.ToTensor()
    ])
    return transform(data), self.get_label_int_by_id(id)
