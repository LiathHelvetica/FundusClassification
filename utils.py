from pandas import read_csv, DataFrame


def flatten(l):
  return [item for sublist in l for item in sublist]


def get_id_from_file_name(f_name: str) -> int:
  return int(f_name.split("-")[-1].split(".")[0])


def get_id_from_f_name(f_name: str) -> str:
  return f_name.split("-")[-1].split(".")[0]


def get_class_for_id(id: str, df: DataFrame) -> str:
  return df[df["ID"] == id]["Disease"].values[0]


def update_resizing(l: list, x_size: int, y_size: int) -> None:
  for dataset in l:
    dataset.x_size = x_size
    dataset.y_size = y_size


def lthv_read_csv(path, fid, sep="-"):
  df = read_csv(path)
  df.index = df.index.to_series().map(lambda i: f"{fid}{sep}{i}")
  return df


def try_or_else(getter, default):
  try:
    return getter()
  except:
    return default
