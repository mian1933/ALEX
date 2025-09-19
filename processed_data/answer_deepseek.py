import re
import os
import time
import json
import pandas as pd
from openai import OpenAI
from ...infer.retrive import query as retrieve_documents

DATA_FOLDER = "[PATH_TO_YOUR_DATA_FOLDER]"
EVALUATION_FOLDER = "[PATH_TO_YOUR_EVALUATION_FOLDER]"
os.makedirs(EVALUATION_FOLDER, exist_ok=True)

client = OpenAI(
    api_key="[YOUR_API_KEY]",
    base_url="[API_ENDPOINT_URL]"
)


def run_llm(prompt, temperature=0.7, max_tokens=512, model="[MODEL_NAME]"):
    system_message = "You are a helpful assistant"
    messages = [{"role": "system", "content": system_message}, {"role": "user", "content": prompt}]
    for _ in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM request failed: {e}, retrying in 2 seconds...")
            time.sleep(2)
    return ""


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
                full_context = f"Background: {prompt_background}\n\nQuestion: {question}"
            else:
                full_context = f"Question: {question}"

            search_query = f"{prompt_background} {question}"
            retrieved_docs = retrieve_documents(search_query, top_k=1)

            prompt = ""
            if retrieved_docs:
                retrieved_content = retrieved_docs[0]['content']
                print(f" [Info] idx: {idx} - Retrieval successful, using RAG mode.")
                prompt = f"""You are an AI legal assistant capable of answering law-related questions.
--- Here is a relevant legal statute for your reference ---
{retrieved_content}
---
Based on the information above and your own knowledge, please answer the following question. You must provide the final answer letter (A/B/C/D) at the end.
{full_context}
Options:
A: {A}
B: {B}
C: {C}
D: {D}
The answer is:"""

            llm_output = run_llm(prompt)

            match = re.findall(r'[A-Da-d]', llm_output)
            answer = match[-1].upper() if match else ""
            answers.append(answer)
            corrects.append(correct)
            idxs.append(idx)

    df = pd.DataFrame({'idx': idxs, 'deepseek_answer': answers, 'correct_answer': corrects})
    output_file = os.path.join(EVALUATION_FOLDER, 'bar_exam_results.csv')
    df.to_csv(output_file, index=False)
    print(f"✅ Multiple-choice questions processed. Results saved to: {output_file}")


def parse_housing_yn(file_path):
    answers, corrects, idxs = [], [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            question = data["question"]
            correct = data["answer"]
            idx = data["idx"]
            state = data.get("state", "").strip()

            search_query = f"{state} {question}"
            retrieved_docs = retrieve_documents(search_query, top_k=1)

            prompt = ""
            if retrieved_docs:
                retrieved_content = retrieved_docs[0]['content']
                print(f" [Info] idx: {idx} - Retrieval successful, using RAG mode.")
                state_clause = f"This question pertains to the law in {state}." if state else ""
                prompt = f"""You are a legal AI assistant capable of answering housing-related questions based on US state law. {state_clause}
--- Here is a relevant legal statute for your reference ---
{retrieved_content}
---
Based on the information above and your own knowledge, please answer the question. You must provide 'yes' or 'no' as the final answer.
Question: {question}
The answer is:"""

            llm_output = run_llm(prompt)

            match = re.findall(r'\b(?:yes|no)\b', llm_output.lower())
            answer = match[-1].upper() if match else ""
            correct = 'YES' if 'yes' in correct.lower() else 'NO'

            answers.append(answer)
            corrects.append(correct)
            idxs.append(idx)

    df = pd.DataFrame({'idx': idxs, 'deepseek_answer': answers, 'correct_answer': corrects})
    output_file = os.path.join(EVALUATION_FOLDER, os.path.basename(file_path).replace('.jsonl', '_results.csv'))
    df.to_csv(output_file, index=False)
    print(f"✅ Yes/No question file {os.path.basename(file_path)} processed. Results saved to: {output_file}")


if __name__ == "__main__":
    print("--- Starting to process Bar Exam MCQs ---")
    parse_bar_exam(os.path.join(DATA_FOLDER, '[BAR_EXAM_FILENAME].jsonl'))

    print("\n--- Starting to process Housing Yes/No Questions ---")
    parse_housing_yn(os.path.join(DATA_FOLDER, '[HOUSING_DATA_FILENAME].jsonl'))

    print("\nAll tasks completed.")