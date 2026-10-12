from src.data_loader.load_entries_summarizer import *
from src.entries_sumamrizer.preprocess_entry import *
def main():
    raw_datasets = load_data()
    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

    model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=4, 
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=5,      
        predict_with_generate=True,
        fp16=True if torch.cuda.is_available() else False,
        logging_steps=100,
        report_to="none"        
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("--- Đang huấn luyện Entry Summarizer... ---")
    trainer.train()

    model.save_pretrained(f"{OUTPUT_DIR}_final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}_final")
    print(f"Hoàn thành! Model đã lưu tại {OUTPUT_DIR}_final")

if __name__ == "__main__":
    main()