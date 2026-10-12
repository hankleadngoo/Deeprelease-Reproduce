from src.data_loader.load_categories import *
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
import evaluate
import numpy as np

df, label_map, encoder = load_category_tsv("E:/Code summerization/Deeprelease Reproduce/data/categories_change/origin_train.tsv")
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)


train_ds = convert_to_hf_dataset(df, tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=len(label_map),
    id2label=label_map,
    label2id={v: k for k, v in label_map.items()}
)


acc_metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return acc_metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(
    output_dir="./deeprelease_category_checkpoints",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=10,
    eval_strategy="no", 
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

if __name__ == "__main__":
    trainer.train()
    model.save_pretrained("E:/Code summerization/Deeprelease Reproduce/data/categories_model_save")
    tokenizer.save_pretrained("E:/Code summerization/Deeprelease Reproduce/data/categories_model_save")
    print("Xong Stage 2: Category Model đã sẵn sàng!")