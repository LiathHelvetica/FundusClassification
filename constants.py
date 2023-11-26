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

ALL_DATA_PATHS = [VALIDATION_DATA_PATH, TEST_DATA_PATH, TRAIN_DATA_PATH]
ALL_LABELS_PATHS = [VALIDATION_LABELS_PATH, TEST_LABELS_PATH, TRAIN_LABELS_PATH]

OUT_PATH = "output"

EXCLUDED_LABELS = ['ERM', 'MS', 'CSR', 'CRVO', 'TV', 'AH', 'ST', 'AION', 'PT', 'RT', 'CRS',
                   'EDN', 'RPEC', 'MHL', 'RP', 'CWS', 'CB', 'ODPM', 'PRH', 'MNF', 'HR',
                   'CRAO', 'TD', 'CME', 'PTCR', 'CF', 'VH', 'MCA', 'VS', 'BRAO', 'PLQ',
                   'HPED', 'CL']

BATCH_SIZES = [4, 10, 16, 20, 25, 30, 35, 50]
