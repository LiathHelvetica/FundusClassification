from dataset import FundusImageDataset


def flatten(l):
  return [item for sublist in l for item in sublist]


def get_id_from_file_name(f_name: str) -> int:
  return int(f_name.split("-")[-1].split(".")[0])


def update_resizing(l: list[FundusImageDataset], x_size: int, y_size: int) -> None:
  for dataset in l:
    dataset.x_size = x_size
    dataset.y_size = y_size
