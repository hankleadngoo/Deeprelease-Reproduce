import torch
import pandas as pd
from transformers import BartForConditionalGeneration, BartTokenizer
import evaluate 
from tqdm import tqdm

model_path = "E:/Code summerization/Deeprelease Reproduce/data/src/data_loader_final"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = BartTokenizer.from_pretrained(model_path)
model = BartForConditionalGeneration.from_pretrained(model_path).to(device)
test_df = pd.read_csv("E:/Code summerization/Deeprelease Reproduce/data/Entry Summarizer/test.csv")

rouge = evaluate.load("rouge")

def generate_summary(text):
    inputs = tokenizer(text, max_length=512, truncation=True, return_tensors="pt").to(device)
    summary_ids = model.generate(
        inputs["input_ids"], 
        max_length=64, 
        num_beams=4, 
        length_penalty=2.0, 
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

predictions = []
references = test_df['abstract'].astype(str).tolist()

print(f"Đang chạy dự đoán trên {len(test_df)} mẫu tập Test...")
for article in tqdm(test_df['article'].astype(str).tolist()):
    predictions.append(generate_summary(article))

results = rouge.compute(predictions=predictions, references=references)

print("\n--- KẾT QUẢ REPRODUCE DEEPRELEASE ---")
print(f"ROUGE-1: {results['rouge1']*100:.2f}")
print(f"ROUGE-2: {results['rouge2']*100:.2f}")
print(f"ROUGE-L: {results['rougeL']*100:.2f}")