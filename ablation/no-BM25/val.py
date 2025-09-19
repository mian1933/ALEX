import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# 读取 CSV 文件
df = pd.read_csv("/home/sa/bar-exam-housing/infer/results/housing_aux/housing_aux_answer_final.csv")   # 修改成你的文件名

# 取出预测和真实标签
y_true = df["correct_answer"].astype(str).tolist()
y_pred = df["deepseek_answer"].astype(str).tolist()

# 计算准确率
accuracy = accuracy_score(y_true, y_pred)

# 计算宏平均 F1
macro_f1 = f1_score(y_true, y_pred, average="macro")

print(f" (Accuracy): {accuracy:.4f}")
print(f" F1 (Macro-F1): {macro_f1:.4f}")
