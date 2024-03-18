RESOURCE_PATH = "data"

VALIDATION_PATH = f"{RESOURCE_PATH}/Evaluation_Set/Evaluation_Set"
VALIDATION_DATA_PATH = f"{VALIDATION_PATH}/Validation"
VALIDATION_LABELS_PATH = f"{VALIDATION_PATH}/RFMiD_Validation_Labels.csv"

TEST_PATH = f"{RESOURCE_PATH}/Test_Set/Test_Set"
TEST_DATA_PATH = f"{TEST_PATH}/Test"
TEST_LABELS_PATH = f"{TEST_PATH}/RFMiD_Testing_Labels.csv"

TRAIN_PATH = f"{RESOURCE_PATH}/Training_Set/Training_Set"
TRAIN_DATA_PATH = f"{TRAIN_PATH}/Training"
TRAIN_LABELS_PATH = f"{TRAIN_PATH}/RFMiD_Training_Labels.csv"

ALL_PATH = f"{RESOURCE_PATH}/All"
ALL_LABEL_PATH = f"{ALL_PATH}/labels.csv"
ALL_DATA_PATH = f"{ALL_PATH}/Data"
ALL_CROPPED_DATA_PATH = f"{ALL_PATH}/DataCropped"
ALL_CROPPED_SQUARE_DATA_PATH = f"{ALL_PATH}/DataCroppedSquare"

OUT_PATH = "output"
TEST_OUT_PATH = f"{OUT_PATH}/test"
TRAIN_OUT_PATH = f"{OUT_PATH}/train"
VALIDATION_OUT_PATH = f"{OUT_PATH}/valid"
AUGMENTATION_OUT_PATH = f"{OUT_PATH}/all"

LABEL_DICT_OUT_PATH = f"{OUT_PATH}/label_dict.json"

VALIDATION_AUGMENT_PATH = VALIDATION_OUT_PATH
TEST_AUGMENT_PATH = TEST_OUT_PATH
TRAIN_AUGMENT_PATH = TRAIN_OUT_PATH

VALIDATION_224_AUGMENT_PATH = f"{OUT_PATH}/224/valid"
TEST_224_AUGMENT_PATH = f"{OUT_PATH}/224/test"
TRAIN_224_AUGMENT_PATH = f"{OUT_PATH}/224/train"

EXCLUDED_LABELS = ['ERM', 'MS', 'CSR', 'CRVO', 'TV', 'AH', 'ST', 'AION', 'PT', 'RT', 'CRS',
                   'EDN', 'RPEC', 'MHL', 'RP', 'CWS', 'CB', 'ODPM', 'PRH', 'MNF', 'HR',
                   'CRAO', 'TD', 'CME', 'PTCR', 'CF', 'VH', 'MCA', 'VS', 'BRAO', 'PLQ',
                   'HPED', 'CL', 'RS', 'LS', 'ODP']

HEALTHY_LABEL = "OK"

CSV_HEADERS = ["acc_val", "epochs", "criterion", "optimizer", "lr", "optimizer-momentum", "weights",
               "scheduler", "scheduler-step-size", "scheduler-gamma", "duration", "loss", "val-size",
               "corrects-total-val", "counters-val"]

TRAIN_DATA_OUT_FILE = f"{OUT_PATH}/train_data.csv"

BATCH_SIZES = [32] # 32, 16
EPOCHS = 1
