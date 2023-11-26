def flatten(l):
  return [item for sublist in l for item in sublist]


def get_id_from_file_name(f_name: str) -> int:
  return int(f_name.split("-")[-1].split(".")[0])
