import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset

def load_category_tsv(file_path):
    df = pd.read_csv(file_path, sep='\t', header=None, names=['label', 'text'])
    df = df.dropna()
    
    encoder = LabelEncoder()
    df['label_id'] = encoder.fit_transform(df['label'])
    
    label_map = {i: label for i, label in enumerate(encoder.classes_)}
    
    return df, label_map, encoder

def convert_to_hf_dataset(df, tokenizer):
    dataset = Dataset.from_pandas(df[['text', 'label_id']].rename(columns={'label_id': 'label'}))
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)
    
    return dataset.map(tokenize_function, batched=True)