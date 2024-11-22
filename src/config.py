DATA_INPUT    = "../input/"
MODEL_OUTPUT  = "../models/"
FILES_OUTPUT  = "../artifacts/"

ORIG_TRAIN_FILE = "../input/train.csv"
TRAINING_FILE   = "../input/train_folds.csv"
TEST_FILE       = "../input/test.csv"
SAMPLE_FILE     = "../input/sample_submission.csv"

N_FOLDS = 5
MODEL   = "logistic"
TARGET_COL   = "Depression"
COLS_TO_DROP = ["Name", "City", "Study Satisfaction", "Academic Pressure", "CGPA", "Profession", "Work Pressure", "Job Satisfaction", "Dietary Habits", "Financial Stress", "Degree", "Sleep Duration", "Dietary Habits", ]
