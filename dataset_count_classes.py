from pandas import read_csv
import os

from constants import OUT_PATH, ALL_LABEL_PATH
from utils import get_class_for_id, get_id_from_f_name

ALL_OUT_PATH = f"{OUT_PATH}/all224"
label_df = read_csv(ALL_LABEL_PATH)

if __name__ == "__main__":
  out = {}
  for f_name in os.listdir(ALL_OUT_PATH):
    id = get_id_from_f_name(f_name)
    cls = get_class_for_id(id, label_df)
    v_or_none = out.get(cls)
    if v_or_none is None:
      out[cls] = 1
    else:
      out[cls] = v_or_none + 1
  print(out)
