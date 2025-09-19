import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics import accuracy_score
import pandas as pd
from tqdm import tqdm

MODEL_PATH = "[PATH_TO_YOUR_TRAINED_MODEL]"
DATA_PATHS = [
    "[PATH_TO_VALIDATION_DATA_1].csv",
    "[PATH_TO_VALIDATION_DATA_2].csv"
]
OUTPUT_PATH = "[PATH_TO_ACCURACY_SUMMARY_CSV]"

label_map = {"simple": 1, "medium": 2, "complex": 3}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

def predict_and_evaluate(df):
    true_labels = df["label"].tolist()
    texts = df["question_text"].tolist()

    pred_labels = []
    for text in tqdm(texts, desc="Predicting"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        output = model.generate(**inputs, max_new_tokens=64)
        pred_str = tokenizer.decode(output[0], skip_special_tokens=True).strip().lower()
        pred_id = label_map.get(pred_str)
        pred_labels.append(pred_id)

    overall_acc = accuracy_score(true_labels, pred_labels)

    per_class_acc = {}
    for cat, cat_id in label_map.items():
        indices = [i for i, t in enumerate(true_labels) if t == cat_id]
        if len(indices) == 0:
            acc_cat = None
        else:
            correct = sum(1 for i in indices if pred_labels[i] == cat_id)
            acc_cat = correct / len(indices)
        per_class_acc[cat] = acc_cat

    return overall_acc, per_class_acc

results = []
for path in DATA_PATHS:
    df = pd.read_csv(path)
    overall_acc, per_class_acc = predict_and_evaluate(df)

    result = {
        "Dataset": path.split("/")[-1],
        "Overall_Accuracy": overall_acc,
        "Simple_Accuracy": per_class_acc["simple"],
        "Medium_Accuracy": per_class_acc["medium"],
        "Complex_Accuracy": per_class_acc["complex"]
    }
    results.append(result)
    print(f"✅ Results for {path}: {result}")

summary_df = pd.DataFrame(results)
summary_df.to_csv(OUTPUT_PATH, index=False)
print(f"🎯 Summary of accuracies saved to: {OUTPUT_PATH}")