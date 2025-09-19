import torch, pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics import accuracy_score
from tqdm import tqdm

model_path = "path/to/final_model"
data_path = "path/to/val_data.csv"
output_path = "path/to/save_results.csv"

label_map = {"simple": 1, "medium": 2, "complex": 3}
id2label = {v: k for k, v in label_map.items()}

df = pd.read_csv(data_path)
true_labels = df["label"].tolist()
texts = df["question_text"].tolist()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)

pred_labels = []
for text in tqdm(texts, desc="Predicting"):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    output = model.generate(**inputs, max_new_tokens=64)
    pred_str = tokenizer.decode(output[0], skip_special_tokens=True).strip().lower()
    pred_id = label_map.get(pred_str)
    pred_labels.append(pred_id)

acc = accuracy_score(true_labels, pred_labels)
print(f"Overall Accuracy: {acc:.4f}")
