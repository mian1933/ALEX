import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset, concatenate_datasets
import copy

OUTPUT_DIR = "[PATH_TO_YOUR_OUTPUT_FOLDER]"
LABELED_PATH = "[PATH_TO_YOUR_LABELED_DATA_CSV]"
UNLABELED_PATH = "[PATH_TO_YOUR_UNLABELED_DATA_FOLDER]"
MODEL_PATH = "[PATH_TO_YOUR_BASE_MODEL]"
CONFIDENCE_THRESHOLD = 0.7
MAX_ITER = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

df = pd.read_csv(LABELED_PATH, sep=",", encoding="utf-8")
print(f"Original data rows: {len(df)}")

def map_label(label):
    mapping = {1: "simple", 2: "medium", 3: "complex"}
    try:
        return mapping[int(label)]
    except (ValueError, TypeError, KeyError):
        return None

df["label"] = df["label"].apply(map_label)
df["text"] = df["question_text"]
df = df[["text", "label"]].dropna().reset_index(drop=True)
print(f"Processed data rows: {len(df)}")

if len(df) == 0:
    raise ValueError("Dataset is empty!")

def tokenize_fn(example):
    if not example["text"] or not example["label"] or not isinstance(example["text"], str) or not isinstance(example["label"], str):
        return None
    try:
        inputs = tokenizer(example["text"], padding="max_length", truncation=True, max_length=512, return_tensors="np")
        labels = tokenizer(text_target=example["label"], padding="max_length", truncation=True, max_length=10, return_tensors="np")
        if not inputs["input_ids"].size or not labels["input_ids"].size or np.all(labels["input_ids"] == tokenizer.pad_token_id):
            return None
        return {
            "input_ids": inputs["input_ids"].flatten(),
            "attention_mask": inputs["attention_mask"].flatten(),
            "labels": labels["input_ids"].flatten(),
        }
    except Exception:
        return None

labeled_dataset = Dataset.from_pandas(df)
labeled_dataset = labeled_dataset.map(tokenize_fn, batched=False, remove_columns=["text", "label"])
labeled_dataset = labeled_dataset.filter(lambda x: x is not None and all(v is not None for v in x.values()))
print(f"Tokenized labeled dataset size: {len(labeled_dataset)}")

train_datasets = copy.deepcopy(labeled_dataset)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    save_strategy="no",
    logging_steps=50,
    report_to="none",
    remove_unused_columns=False,
)

for iteration in range(MAX_ITER):
    print(f"\n=== Pseudo-Labeling Iteration {iteration + 1}/{MAX_ITER} ===")

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to("cuda" if torch.cuda.is_available() else "cpu")

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_datasets,
        data_collator=data_collator,
    )

    trainer.train()

    new_pseudo = []
    for filename in os.listdir(UNLABELED_PATH):
        if not filename.endswith(".csv"):
            continue
        filepath = os.path.join(UNLABELED_PATH, filename)
        try:
            unlabeled_df = pd.read_csv(filepath)
            for _, row in unlabeled_df.iterrows():
                text = row.get("question_text")
                if not text or not isinstance(text, str) or not text.strip():
                    continue
                inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        output_scores=True,
                        return_dict_in_generate=True,
                        max_new_tokens=10,
                    )
                pred_str = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True).strip().lower()
                if pred_str not in ["simple", "medium", "complex"]:
                    continue
                if outputs.scores and torch.nn.functional.softmax(outputs.scores[0], dim=-1).max().item() >= CONFIDENCE_THRESHOLD:
                    new_pseudo.append({"text": text.strip(), "label": pred_str})
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            continue

    if not new_pseudo:
        print("No new pseudo-labels generated. Stopping training.")
        break

    print(f"Generated {len(new_pseudo)} new pseudo-labels.")

    pseudo_dataset = Dataset.from_list(new_pseudo)
    pseudo_dataset = pseudo_dataset.map(tokenize_fn, batched=False, remove_columns=["text", "label"])
    pseudo_dataset = pseudo_dataset.filter(lambda x: x is not None and all(v is not None for v in x.values()))
    print(f"Tokenized pseudo-label dataset size: {len(pseudo_dataset)}")

    train_datasets = concatenate_datasets([labeled_dataset, pseudo_dataset])

model.save_pretrained(os.path.join(OUTPUT_DIR, "[FINAL_MODEL_NAME]"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "[FINAL_MODEL_NAME]"))
print(f"Model saved to {os.path.join(OUTPUT_DIR, '[FINAL_MODEL_NAME]')}")

final_df = pd.DataFrame([
    {"text": tokenizer.decode(sample["input_ids"], skip_special_tokens=True),
     "label": tokenizer.decode(sample["labels"], skip_special_tokens=True)}
    for sample in labeled_dataset
])
final_df.to_csv(os.path.join(OUTPUT_DIR, "[FINAL_DATASET_NAME].csv"), index=False)
print(f"Final labeled data saved to [FINAL_DATASET_NAME].csv")