import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    BartForConditionalGeneration, 
    BartTokenizer, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer, 
    DataCollatorForSeq2Seq
)
ROOT = "E:/Code summerization/Deeprelease Reproduce/data"
TRAIN_FILE = f"{ROOT}/Entry Summarizer/train.csv"
VAL_FILE = f"{ROOT}/Entry Summarizer/valid.csv"
TEST_FILE = f"{ROOT}/Entry Summarizer/test.csv"
MODEL_NAME = "facebook/bart-base"
OUTPUT_DIR = f"{ROOT}/src/entries_summarizer_model_save"

def load_data():
    def read_csv(file):
        df = pd.read_csv(file)
        df = df.dropna(subset=['article', 'abstract'])
        return df

    train_df = read_csv(TRAIN_FILE)
    val_df = read_csv(VAL_FILE)
    test_df = read_csv(TEST_FILE)

    raw_datasets = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "validation": Dataset.from_pandas(val_df),
        "test": Dataset.from_pandas(test_df)
    })
    return raw_datasets
