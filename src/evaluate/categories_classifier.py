import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

model_path = "E:/Code summerization/Deeprelease Reproduce/data/categories_model_save"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

test_df = pd.read_csv("E:/Code summerization/Deeprelease Reproduce/data/categories_change/test.tsv", sep='\t', header=None, names=['label', 'text'])

target_names = sorted(test_df['label'].unique().tolist())
label2id = {label: i for i, label in enumerate(target_names)}

y_true = []
y_pred = []

print("Đang đánh giá Category Model...")
for _, row in test_df.iterrows():
    inputs = tokenizer(row['text'], return_tensors="pt", truncation=True, padding=True, max_length=64).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    prediction = torch.argmax(outputs.logits, dim=-1).item()
    
    y_true.append(label2id[row['label']])
    y_pred.append(prediction)

print("\n--- BÁO CÁO PHÂN LOẠI (DETAILED REPORT) ---")
print(classification_report(y_true, y_pred, target_names=target_names))