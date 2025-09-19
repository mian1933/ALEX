import json
import re
import os
import pandas as pd
from openai import OpenAI
import time
from tqdm import tqdm
from typing import Dict, Any, Optional, List
import argparse
import csv

print("Loading dependencies...")
from retrive import query as retrieve_documents
from Determine_complexity import classify_multiple_choice_complexity, classify_yes_no_complexity

print("✅ All dependencies loaded successfully.\n")

client = OpenAI(
    api_key="[YOUR_API_KEY]",
    base_url="[API_ENDPOINT_URL]"
)

SELF_CRITIQUE_PROMPT_TEMPLATE = "[A template for a senior agent to critique a junior's answer and identify specific information needed for verification.]"
COMPLEX_SYNTHESIS_PROMPT_TEMPLATE = "[A template that guides a step-by-step reasoning and verification process to synthesize all information into a final, conclusive answer.]"

def run_llm(prompt: str, temperature: float = 0.1, max_tokens: int = 500, model: str = "[MODEL_NAME]"):
    system_message = "[Instruction for the AI to act as a precise legal assistant, analyzing context and providing a formatted answer.]"
    messages = [{"role": "system", "content": system_message}, {"role": "user", "content": prompt}]
    for _ in range(3):
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, temperature=temperature,
                max_tokens=max_tokens, stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            time.sleep(2)
    return ""


def _build_simple_rag_prompt(prompt_type: str, prompt_data: Dict[str, Any], retrieved_doc: Dict[str, Any]) -> str:
    retrieved_content_str = f"--- Relevant Legal Statute ---\n{retrieved_doc['content']}"
    if prompt_type == 'choice':
        return f"""[Instruction for the AI to answer a multiple-choice question using the provided statute.]
{retrieved_content_str}
---
{prompt_data['full_context']}
Choices: A, B, C, D...
Answer:"""
    elif prompt_type == 'yes_no':
        return f"""[Instruction for the AI to answer a yes/no question using the provided statute.]
{retrieved_content_str}
---
Question: {prompt_data['question']}
Answer:"""
    return ""


def _build_refine_prompt(prompt_type: str, prompt_data: Dict[str, Any], existing_answer: str,
                         new_context_doc: Dict[str, Any]) -> str:
    new_context = f"--- Additional Context ---\n{new_context_doc['content']}"
    question_context = prompt_data.get('full_context') or prompt_data.get('question')
    return f"""[Instruction to refine an existing answer based on new context.]
Original Question: {question_context}
Existing Answer: "{existing_answer}"
New Context: {new_context}
---
Refined Answer:"""


def _handle_medium_mode(prompt_type: str, prompt_data: Dict[str, Any]) -> str:
    print(f"  [Medium Mode] Initiating 2-step refinement process...")
    if prompt_type == 'choice':
        initial_query = prompt_data.get('full_context', '')
    else:
        initial_query = f"{prompt_data.get('state', '')} {prompt_data.get('question', '')}"

    retrieved_docs_1 = retrieve_documents(initial_query, top_k=1)
    if not retrieved_docs_1: return ""

    llm_output_1 = run_llm(_build_simple_rag_prompt(prompt_type, prompt_data, retrieved_docs_1[0]))
    if not llm_output_1: return ""

    refine_query = f"Original question: {initial_query}\nInitial retrieved context: {retrieved_docs_1[0]['content']}\nInitial answer: {llm_output_1}"
    retrieved_docs_2 = retrieve_documents(refine_query, top_k=1)

    if retrieved_docs_2:
        return run_llm(_build_refine_prompt(prompt_type, prompt_data, llm_output_1, retrieved_docs_2[0]))
    else:
        return llm_output_1


def _handle_complex_mode(prompt_type: str, prompt_data: Dict[str, Any]) -> str:
    print(f"  [Complex Mode] Initiating self-critique and synthesis process...")
    if prompt_type == 'choice':
        initial_query = prompt_data.get('full_context', '')
    else:
        initial_query = f"{prompt_data.get('state', '')} {prompt_data.get('question', '')}"

    initial_docs = retrieve_documents(initial_query, top_k=1)
    if not initial_docs: return ""

    initial_context = initial_docs[0]['content']
    preliminary_answer = run_llm(_build_simple_rag_prompt(prompt_type, prompt_data, initial_docs[0]))
    if not preliminary_answer: return ""

    needed_verification = run_llm(
        SELF_CRITIQUE_PROMPT_TEMPLATE.format(question_context=initial_query, initial_context=initial_context,
                                             preliminary_answer=preliminary_answer),
        max_tokens=200
    )
    if not needed_verification: return ""

    additional_docs = retrieve_documents(
        f"To verify the answer for '{initial_query}', I need specific information about: {needed_verification}",
        top_k=5
    )
    if not additional_docs: return ""

    additional_context = f"--- Additional Context ---\n{additional_docs[0]['content']}"

    if prompt_type == 'choice':
        final_question_body = f"{prompt_data['full_context']}\nChoices:\nA: {prompt_data['A']}\nB: {prompt_data['B']}\nC: {prompt_data['C']}\nD: {prompt_data['D']}"
    else:
        state_clause = f"The question pertains to the law in {prompt_data['state']}." if prompt_data.get(
            'state') else ""
        final_question_body = f"{state_clause}\nQuestion: {prompt_data['question']}"

    return run_llm(
        COMPLEX_SYNTHESIS_PROMPT_TEMPLATE.format(question_context=final_question_body, initial_context=initial_context,
                                                 additional_context=additional_context)
    )


def process_bar_exam(file_path, output_folder):
    print(f"\n--- Starting to process Multiple-Choice Question file: {file_path} ---")
    output_file = os.path.join(output_folder, 'bar_exam_answer_final.csv')
    processed_ids = set()

    if os.path.exists(output_file):
        print(f"Existing results file found: {output_file}. Loading progress...")
        try:
            df_existing = pd.read_csv(output_file, usecols=['idx'], dtype={'idx': str})
            processed_ids = set(df_existing['idx'])
        except (ValueError, pd.errors.EmptyDataError):
            processed_ids = set()

    with open(output_file, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['idx', 'deepseek_answer', 'correct_answer', 'complexity', 'llm_output']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not processed_ids: writer.writeheader()

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            with tqdm(total=len(lines), desc="Processing MCQs", initial=len(processed_ids)) as pbar:
                for line in lines:
                    data = json.loads(line)
                    current_idx = str(data.get('idx', ''))
                    if current_idx in processed_ids:
                        continue

                    question, prompt_background = data["question"], data.get("prompt", "")
                    A, B, C, D = data["choice_a"], data["choice_b"], data["choice_c"], data["choice_d"]
                    correct = data["answer"]
                    complexity = classify_multiple_choice_complexity(background=prompt_background, question=question,
                                                                     choice_a=A, choice_b=B, choice_c=C, choice_d=D)

                    context_prefix = f"Background: {prompt_background}\n\n" if prompt_background and str(
                        prompt_background).strip().lower() != "nan" else ""
                    full_context = f"{context_prefix}Question: {question}"
                    prompt_data = {'full_context': full_context, 'A': A, 'B': B, 'C': C, 'D': D}

                    llm_output = ""
                    if complexity == 'simple':
                        retrieved_docs = retrieve_documents(f"{prompt_background} {question}", top_k=1)
                        if retrieved_docs:
                            llm_output = run_llm(_build_simple_rag_prompt('choice', prompt_data, retrieved_docs[0]))
                    elif complexity == 'medium':
                        llm_output = _handle_medium_mode('choice', prompt_data)
                    elif complexity == 'complex':
                        llm_output = _handle_complex_mode('choice', prompt_data)

                    if not llm_output:
                        print(f"  [Fallback] No RAG output for idx: {current_idx}. Using non-retrieval mode.")
                        llm_output = run_llm(f"[A fallback prompt for MCQs...]\n{full_context}\nChoices...\nAnswer:")

                    found_chars = re.findall(r'[A-Da-d]', llm_output)
                    answer = found_chars[-1].upper() if found_chars else ''

                    writer.writerow({'idx': current_idx, 'deepseek_answer': answer, 'correct_answer': correct,
                                     'complexity': complexity, 'llm_output': llm_output})
                    pbar.update(1)

    print(f"✅ Bar exam processing complete. Results saved to: {output_file}")


def process_housing_yn(file_path, output_folder):
    print(f"\n--- Starting to process Yes/No Question file: {file_path} ---")
    output_file = os.path.join(output_folder, os.path.basename(file_path).replace('.jsonl', '_answer_final.csv'))
    processed_ids = set()

    if os.path.exists(output_file):
        print(f"Existing results file found: {output_file}. Loading progress...")
        try:
            df_existing = pd.read_csv(output_file, usecols=['idx'], dtype={'idx': str})
            processed_ids = set(df_existing['idx'])
        except (ValueError, pd.errors.EmptyDataError):
            processed_ids = set()

    with open(output_file, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['idx', 'deepseek_answer', 'correct_answer', 'complexity', 'llm_output']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not processed_ids: writer.writeheader()

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            with tqdm(total=len(lines), desc="Processing Yes/No Questions", initial=len(processed_ids)) as pbar:
                for line in lines:
                    data = json.loads(line)
                    current_idx = str(data.get('idx', ''))
                    if current_idx in processed_ids:
                        continue

                    question, correct = data["question"], data["answer"]
                    state = data.get("state", "").strip()
                    complexity = classify_yes_no_complexity(state=state, question=question)
                    prompt_data = {'question': question, 'state': state}

                    llm_output = ""
                    if complexity == 'simple':
                        retrieved_docs = retrieve_documents(f"{state} {question}", top_k=1)
                        if retrieved_docs:
                            llm_output = run_llm(_build_simple_rag_prompt('yes_no', prompt_data, retrieved_docs[0]))
                    elif complexity == 'medium':
                        llm_output = _handle_medium_mode('yes_no', prompt_data)
                    elif complexity == 'complex':
                        llm_output = _handle_complex_mode('yes_no', prompt_data)

                    if not llm_output:
                        print(f"  [Fallback] No RAG output for idx: {current_idx}. Using non-retrieval mode.")
                        llm_output = run_llm(
                            f"[A fallback prompt for Yes/No questions...]\nQuestion: {question}\nAnswer:")

                    answer_text = llm_output.lower().rsplit(maxsplit=1)[-1] if llm_output else ''
                    answer = 'YES' if 'yes' in answer_text else ('NO' if 'no' in answer_text else '')

                    writer.writerow({'idx': current_idx, 'deepseek_answer': answer,
                                     'correct_answer': 'YES' if 'yes' in correct.lower() else 'NO',
                                     'complexity': complexity, 'llm_output': llm_output})
                    pbar.update(1)

    print(f"✅ Yes/No processing complete. Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Process legal question files using retrieval and LLMs.",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--bar-exam-file', type=str, help="Path to the multiple-choice (bar exam) JSONL file.")
    parser.add_argument('--housing-file', type=str, help="Path to the yes/no (housing) JSONL file.")
    parser.add_argument('--output-folder', type=str, required=True,
                        help="Path to the folder for storing result CSV files.")

    args = parser.parse_args()
    bar_exam_file, housing_file, evaluate_folder = args.bar_exam_file, args.housing_file, args.output_folder

    if not os.path.exists(evaluate_folder):
        os.makedirs(evaluate_folder)
        print(f"Created output folder: {evaluate_folder}")

    if bar_exam_file and os.path.exists(bar_exam_file):
        process_bar_exam(bar_exam_file, evaluate_folder)
    elif bar_exam_file:
        print(f"Error: File not found {bar_exam_file}, skipping task.")

    if housing_file and os.path.exists(housing_file):
        process_housing_yn(housing_file, evaluate_folder)
    elif housing_file:
        print(f"Error: File not found {housing_file}, skipping task.")

    if not bar_exam_file and not housing_file:
        print("Warning: No input files were provided. Exiting program.")

    print("\nAll tasks completed.")


if __name__ == '__main__':
    main()