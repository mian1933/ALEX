import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_csv("/home/sa/bar-exam-housing/infer/results/housing_aux/housing_aux_answer_final.csv")

y_true = df["correct_answer"].astype(str).tolist()
y_pred = df["deepseek_answer"].astype(str).tolist()

accuracy = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(y_true, y_pred, average="macro")
weighted_f1 = f1_score(y_true, y_pred, average="weighted")

print(f"Accuracy: {accuracy:.4f}")
print(f"Macro-F1: {macro_f1:.4f}")
print(f"Weighted-F1: {weighted_f1:.4f}")
