from src.data_loader.load_entries_summarizer import *

tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
def preprocess_function(examples):
    inputs = [doc for doc in examples["article"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["abstract"], max_length=128, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs       

#$env:PYTHONPATH = "."
#python src/entries_sumamrizer/preprocess_entry.py