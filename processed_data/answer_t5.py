import os
import re
import time
import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

DATA_FOLDER = "[PATH_TO_YOUR_LABELED_DATA]"
EVALUATION_FOLDER = "[PATH_TO_YOUR_EVALUATION_OUTPUT]"
MODEL_PATH = "[PATH_TO_YOUR_T5_MODEL]"

os.makedirs(EVALUATION_FOLDER, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to("cuda" if torch.cuda.is_available() else "cpu")

def run_t5(prompt, max_tokens=64):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_tokens)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("T5 answer:", decoded)
    return decoded

def parse_bar_exam(file_path):
    answers, corrects, idxs = [], [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            question = data["question"]
            prompt_background = data.get("prompt", "")
            A, B, C, D = data["choice_a"], data["choice_b"], data["choice_c"], data["choice_d"]
            correct = data["answer"]
            idx = data["idx"]

            if prompt_background and str(prompt_background).strip().lower() != "nan":
                full_context = f"Background information: {prompt_background}\n\nQuestion: {question}"
            else:
                full_context = f"Question: {question}"

            prompt = f"""You are an AI legal assistant capable of answering law-related questions.
The following is a single-choice question. Please answer with only the letter (A/B/C/D) representing the correct choice. Do not explain or repeat the content.
{full_context}
Options:
A: {A}
B: {B}
C: {C}
D: {D}
The answer is:"""

            answer = run_t5(prompt)
            answer = ''.join(re.findall(r'[A-Da-d]', answer)).upper()
            answers.append(answer)
            corrects.append(correct)
            idxs.append(idx)

    df = pd.DataFrame({'idx': idxs, 't5_answer': answers, 'correct_answer': corrects})
    output_file = os.path.join(EVALUATION_FOLDER, 'bar_exam_t5_results.csv')
    df.to_csv(output_file, index=False)

def parse_housing_yn(file_path):
    answers, corrects, idxs = [], [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            question = data["question"]
            correct = data["answer"]
            idx = data["idx"]
            state = data.get("state", "").strip()

            if state:
                prompt = f"""You are a legal AI assistant capable of answering housing-related legal questions in the United States, particularly in {state}.
Please answer only with "Yes" or "No", without any explanation.

Question: {question}
Answer:"""
            else:
                prompt = f"""You are a legal AI assistant capable of answering housing-related legal questions.
Please answer only with "Yes" or "No", without any explanation.

Question: {question}
Answer:"""

            answer = run_t5(prompt)
            answer = 'YES' if 'yes' in answer.lower() else 'NO'
            correct = 'YES' if 'yes' in correct.lower() else 'NO'

            answers.append(answer)
            corrects.append(correct)
            idxs.append(idx)

    df = pd.DataFrame({'idx': idxs, 't5_answer': answers, 'correct_answer': corrects})
    output_file = os.path.join(EVALUATION_FOLDER, os.path.basename(file_path).replace('.jsonl', '_t5_results.csv'))
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    parse_bar_exam(os.path.join(DATA_FOLDER, '[BAR_EXAM_FILENAME].jsonl'))
    parse_housing_yn(os.path.join(DATA_FOLDER, '[HOUSING_DATA_FILENAME].jsonl'))