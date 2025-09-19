import json
import re
import os
import pandas as pd
from openai import OpenAI
import time
from tqdm import tqdm
from typing import Dict, Any
import csv
from .retrive import query as retrieve_documents

client = OpenAI(api_key="...", base_url="...")

def run_llm(prompt: str, temperature: float = ..., max_tokens: int = ..., model: str = "..."):
    """Calls the LLM API with retry logic."""
    system_message = "Instruction: analyze legal context and question, return final concise answer"
    messages = [{"role": "system", "content": system_message}, {"role": "user", "content": prompt}]
    for _ in range(3):
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, temperature=temperature,
                max_tokens=max_tokens, stream=False
            )
            return response.choices[0].message.content
        except Exception:
            time.sleep(2)
    return ""

def _build_simple_rag_prompt(prompt_type: str, prompt_data: Dict[str, Any], retrieved_doc: Dict[str, Any]) -> str:
    """Builds the initial RAG prompt."""
    retrieved_content_str = f"--- Retrieved Context ---\n{retrieved_doc['content']}"
    if prompt_type == 'choice':
        return f"""Choice question with reference:\n{retrieved_content_str}\n{prompt_data['full_context']}\nChoices:\nA: {prompt_data['A']}\nB: {prompt_data['B']}\nC: {prompt_data['C']}\nD: {prompt_data['D']}\nAnswer:"""
    elif prompt_type == 'yes_no':
        state_clause = f"This question pertains to {prompt_data['state']}." if prompt_data.get('state') else ""
        return f"""{state_clause}\nYes/No question with reference:\n{retrieved_content_str}\nQuestion: {prompt_data['question']}\nAnswer:"""
    return ""

def _build_refine_prompt(prompt_type: str, prompt_data: Dict[str, Any], existing_answer: str, new_context_doc: Dict[str, Any]) -> str:
    """Builds a prompt to refine an existing answer."""
    new_context = f"--- Additional Context ---\n{new_context_doc['content']}"
    question_context = prompt_data.get('full_context') or prompt_data.get('question')
    refine_intro = f"""Original question: {question_context}\nExisting answer: "{existing_answer}"\nNew context:\n{new_context}\nRefine prompt."""
    if prompt_type == 'choice':
        return f"{refine_intro} Final answer must be A, B, C, or D.\nRefined prompt:"
    elif prompt_type == 'yes_no':
        state_clause = f"The question pertains to {prompt_data['state']}." if prompt_data.get('state') else ""
        return f"{state_clause}\n{refine_intro} Final answer must be yes or no.\nRefined prompt:"
    return ""

def _handle_medium_mode(prompt_type: str, prompt_data: Dict[str, Any]) -> str:
    """Handles the two-step RAG process for medium complexity questions."""
    if prompt_type == 'choice':
        search_query_1 = prompt_data.get('full_context', '')
    elif prompt_type == 'yes_no':
        question_context = prompt_data.get('question', '')
        state = prompt_data.get('state', '')
        search_query_1 = f"{state} {question_context}"
    else:
        return ""
    retrieved_docs_1 = retrieve_documents(search_query_1, top_k=...)
    if not retrieved_docs_1:
        return ""
    prompt_1 = _build_simple_rag_prompt(prompt_type, prompt_data, retrieved_docs_1[0])
    llm_output_1 = run_llm(prompt_1)
    if not llm_output_1:
        return ""
    search_query_2 = f"Original question: {search_query_1}\nInitial context: {retrieved_docs_1[0]['content']}\nInitial answer: {llm_output_1}"
    retrieved_docs_2 = retrieve_documents(search_query_2, top_k=...)
    if retrieved_docs_2:
        prompt_2 = _build_refine_prompt(prompt_type, prompt_data, llm_output_1, retrieved_docs_2[0])
        return run_llm(prompt_2)
    else:
        return llm_output_1

def process_housing_yn(jsonl_file_path, source_csv_folder, output_folder):
    """
    Processes housing questions.
    MODIFIED: Now re-processes ONLY 'medium' complexity questions and keeps all others.
    """
    source_csv_path = os.path.join(source_csv_folder, '...')
    output_file_path = os.path.join(output_folder, '...')

    questions_from_jsonl = {}
    try:
        with open(jsonl_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                questions_from_jsonl[str(data['idx'])] = data
    except FileNotFoundError:
        print(f"Error: JSONL file not found at {jsonl_file_path}")
        return

    try:
        df_source = pd.read_csv(source_csv_path, dtype={'idx': str})
    except FileNotFoundError:
        print(f"Error: Source CSV not found at {source_csv_path}")
        return

    fieldnames = ['idx', 'deepseek_answer', 'correct_answer', 'llm_output', 'complexity', 'status']

    with open(output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in tqdm(df_source.itertuples(), total=len(df_source), desc="Processing Housing Questions"):
            current_idx = str(row.idx)
            complexity = str(getattr(row, 'complexity', '')).strip()

            # **MODIFIED LOGIC**: If complexity is NOT medium, keep the original row and skip.
            if complexity != 'medium':
                writer.writerow({
                    'idx': current_idx,
                    'deepseek_answer': getattr(row, 'deepseek_answer', ''),
                    'correct_answer': getattr(row, 'correct_answer', ''),
                    'llm_output': getattr(row, 'llm_output', ''),
                    'complexity': complexity,
                    'status': 'KEPT'
                })
                continue

            # The rest of the code now only runs for 'medium' complexity questions.
            if current_idx not in questions_from_jsonl:
                continue

            question_data = questions_from_jsonl[current_idx]
            prompt_data = {
                'question': question_data['question'],
                'state': question_data.get('state', '')
            }

            llm_output = _handle_medium_mode('yes_no', prompt_data)

            if not llm_output:
                state_clause = f"This question pertains to {question_data.get('state')}." if question_data.get('state') else ""
                direct_prompt = f"""{state_clause} Direct yes/no question.\nQuestion: {question_data['question']}\nAnswer:"""
                llm_output = run_llm(direct_prompt)

            answer_text = llm_output.lower().rsplit(maxsplit=1)[-1] if llm_output else ''
            answer = 'YES' if 'yes' in answer_text else ('NO' if 'no' in answer_text else '')
            correct_answer_formatted = 'YES' if 'yes' in question_data.get('answer', '').lower() else 'NO'

            writer.writerow({
                'idx': current_idx,
                'deepseek_answer': answer,
                'correct_answer': correct_answer_formatted,
                'llm_output': llm_output,
                'complexity': 'medium', # Set complexity to medium as it's the one being processed
                'status': 'REPROCESSED'
            })

def main():
    """Main function to run the script."""
    jsonl_file_path = "..."
    source_csv_folder = "..."
    output_folder = "..."
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    process_housing_yn(jsonl_file_path, source_csv_folder, output_folder)

if __name__ == '__main__':
    main()