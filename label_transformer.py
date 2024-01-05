import json

from pandas import concat, factorize

from constants import VALIDATION_LABELS_PATH, TEST_LABELS_PATH, TRAIN_LABELS_PATH, LABEL_DICT_OUT_PATH
from dataset import get_column_name
from utils import lthv_read_csv

df_val = lthv_read_csv(VALIDATION_LABELS_PATH, "val")
df_test = lthv_read_csv(TEST_LABELS_PATH, "tst")
df_train = lthv_read_csv(TRAIN_LABELS_PATH, "trn")

df_all = concat([df_val, df_test, df_train])

df_all["Disease"] = df_all.drop(["ID", "Disease_Risk"], axis=1).apply(get_column_name, axis=1)
diseases = df_all[["Disease"]]
diseases["ID"] = factorize(diseases["Disease"])[0]
diseases = diseases.drop_duplicates()
out = json.dumps(diseases.set_index("Disease")["ID"].to_dict())

with open(LABEL_DICT_OUT_PATH, "w") as f:
  f.write(out)
