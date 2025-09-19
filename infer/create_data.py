import pandas as pd
import csv


def process_file(input_path):
    rows = []
    df = pd.read_parquet(input_path)

    for idx, row in df.iterrows():
        if "choice_a" in row and pd.notna(row["choice_a"]):
            prompt_background = row.get("prompt", "")
            question = row.get("question", "")
            A = row["choice_a"]
            B = row["choice_b"]
            C = row["choice_c"]
            D = row["choice_d"]

            if pd.notna(prompt_background) and str(prompt_background).strip():
                full_context = f"Background：{prompt_background}\n\nquestion：{question}"
            else:
                full_context = f"question：{question}"

            question_text = (
                "Determine the complexity of the following legal multiple-choice question (simple, medium, or complex):\n"
                f"{full_context}\n"
                f"Choices:\nA: {A}\nB: {B}\nC: {C}\nD: {D}\n"
                "Answer:"
            )
        else:  # Yes/No题 (housing)
            state = row.get("state", "")
            question = row.get("question", "")
            full_context = f"state：{state}\n\nquestion：{question}"
            question_text = (
                "Classify the complexity level of the following legal Yes/No question as one of the following: Simple, Medium, or Complex.:\n"
                f"Question：{full_context}\n"
                "Answer:"
            )

        rows.append([question_text])

    return rows



housing_rows = process_file('/home/sa/bar-exam-housing/dataset/HousingQA/questions_aux.parquet')


with open('./complexity_prompt/housing_complexity_prompts.csv', "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["question_text"])
    writer.writerows(housing_rows)


bar_exam_rows = process_file('/home/sa/bar-exam-housing/dataset/Bar-Exam-QA/barexam_qa.parquet')

with open('./complexity_prompt/bar_complexity_prompts.csv', "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["question_text"])
    writer.writerows(bar_exam_rows)



df_bar = pd.read_parquet('../dataset/Bar-Exam-QA/barexam_qa.parquet')
df_housing_aux = pd.read_parquet('../dataset/HousingQA/questions_aux.parquet')


df_bar.to_json('./infer_data/bar_exam.jsonl', orient='records', lines=True)
df_housing_aux.to_json('./infer_data/housing_aux.jsonl', orient='records', lines=True)
