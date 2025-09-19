import os, torch, pandas as pd, numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
import copy

OUTPUT_DIR = "path/to/output"
LABELED_PATH = "path/to/data.csv"
MODEL_PATH = "path/to/model"

os.makedirs(OUTPUT_DIR, exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

df = pd.read_csv(LABELED_PATH)
df["label"] = df["label"].apply(lambda x: {1: "simple", 2: "medium", 3: "complex"}.get(int(x), None))
df["text"] = df["question_text"]
df = df[["text", "label"]].dropna().reset_index(drop=True)

if len(df) == 0:
    raise ValueError("Empty dataset")

def tokenize_fn(example):
    inputs = tokenizer(example["text"], padding="max_length", truncation=True, max_length=512, return_tensors="np")
    labels = tokenizer(text_target=example["label"], padding="max_length", truncation=True, max_length=10, return_tensors="np")
    return {"input_ids": inputs["input_ids"].flatten(),
            "attention_mask": inputs["attention_mask"].flatten(),
            "labels": labels["input_ids"].flatten()}

labeled_dataset = Dataset.from_pandas(df)
labeled_dataset = labeled_dataset.map(tokenize_fn, batched=False, remove_columns=["text", "label"])
labeled_dataset = labeled_dataset.filter(lambda x: x is not None)
train_datasets = copy.deepcopy(labeled_dataset)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=20,
    num_train_epochs=3,
    save_strategy="no",
    logging_steps=50,
    report_to="none",
    remove_unused_columns=False,
)

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to("cuda" if torch.cuda.is_available() else "cpu")
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

trainer = Trainer(model=model, args=training_args, train_dataset=train_datasets, data_collator=data_collator)
trainer.train()

model.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))

final_df = pd.DataFrame([
    {"text": tokenizer.decode(s["input_ids"], skip_special_tokens=True),
     "label": tokenizer.decode(s["labels"], skip_special_tokens=True)}
    for s in labeled_dataset
])
final_df.to_csv(os.path.join(OUTPUT_DIR, "labeled_final.csv"), index=False)
