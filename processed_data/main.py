import pandas as pd

# 加载数据集
df_bar = pd.read_parquet('../dataset/Bar-Exam-QA/barexam_qa.parquet')
df_housing_aux = pd.read_parquet('../dataset/HousingQA/questions_aux.parquet')


RANDOM_SEED = 42


df_bar_labeled = df_bar.sample(frac=0.2, random_state=RANDOM_SEED)
df_housing_aux_labeled = df_housing_aux.sample(frac=0.2, random_state=RANDOM_SEED)


df_bar_labeled.to_json('./labeled/bar_exam_labeled.jsonl', orient='records', lines=True)
df_housing_aux_labeled.to_json('./labeled/housing_aux_labeled.jsonl', orient='records', lines=True)

def split_remaining(df_full, df_labeled):
    df_remaining = df_full.drop(df_labeled.index)
    df_unlabeled = df_remaining.sample(frac=5/8, random_state=RANDOM_SEED)
    df_val = df_remaining.drop(df_unlabeled.index)
    return df_unlabeled, df_val

df_bar_unlabeled, df_bar_val = split_remaining(df_bar, df_bar_labeled)
df_housing_aux_unlabeled, df_housing_aux_val = split_remaining(df_housing_aux, df_housing_aux_labeled)


df_bar_unlabeled.to_json('./unlabeled/bar_exam_unlabeled.jsonl', orient='records', lines=True)
df_housing_aux_unlabeled.to_json('./unlabeled/housing_aux_unlabeled.jsonl', orient='records', lines=True)

df_bar_val.to_json('./val/bar_exam_val.jsonl', orient='records', lines=True)
df_housing_aux_val.to_json('./val/housing_aux_val.jsonl', orient='records', lines=True)


import json
import csv
import os


input_files_unlabeled = {
    "housing_aux_unlabeled.jsonl": "housing_aux_unlabeled.csv",
    "housing_unlabeled.jsonl": "housing_unlabeled.csv",
    "bar_exam_unlabeled.jsonl": "bar_exam_unlabeled.csv"
}
input_files_val = {
    "housing_aux_val.jsonl": "housing_aux_val.csv",
    "housing_val.jsonl": "housing_val.csv",
    "bar_exam_val.jsonl": "bar_exam_val.csv"
}

def process_file(input_path, output_path):
    rows = []
    with open(input_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            data = json.loads(line.strip())
            if "choice_a" in data:  # 多选题（bar_exam）
                prompt_background = data.get("prompt", "")
                question = data.get("question", "")
                A = data["choice_a"]
                B = data["choice_b"]
                C = data["choice_c"]
                D = data["choice_d"]
                if prompt_background and str(prompt_background).strip().lower() != "nan":
                    full_context = f"Background：{prompt_background}\n\nquestion：{question}"
                else:
                    full_context = f"question：{question}"
                question_text = (
                    "Determine the complexity of the following legal multiple-choice question (simple, medium, or complex):\n"
                    f"{full_context}\n"
                    f"Choices:\nA: {A}\nB: {B}\nC: {C}\nD: {D}\n"
                    "Answer:"
                )
            else:  # Yes/No题（housing）
                state = data.get("state", "")
                question = data.get("question", "")
                full_context = f"state：{state}\n\nquestion：{question}"
                question_text = (
                    "Classify the complexity level of the following legal Yes/No question as one of the following: Simple, Medium, or Complex.:\n"
                    f"Question：{full_context}\n"
                    "Answer:"
                )
            rows.append([question_text])

    # 写入 CSV
    with open(output_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["question_text"])
        writer.writerows(rows)



for jsonl_file, csv_file in input_files_unlabeled.items():
    input_path = os.path.join("./unlabeled", jsonl_file)
    output_path = os.path.join("./unlabeled", csv_file)
    process_file(input_path, output_path)

for jsonl_file, csv_file in input_files_val.items():
    input_path = os.path.join("..","val", jsonl_file)
    output_path = os.path.join("..","val", csv_file)



    process_file(input_path, output_path)
print("转换完成！")
