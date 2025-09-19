import os
import time
import json
import pandas as pd
import textstat
import re
from openai import OpenAI
from typing import List, Dict, Any, Callable
import random

data_folder = "[PATH_TO_YOUR_DATA_FOLDER]"
evaluate_folder = "[PATH_TO_YOUR_EVALUATION_FOLDER]"
os.makedirs(evaluate_folder, exist_ok=True)

client = OpenAI(
    api_key="[YOUR_API_KEY]",
    base_url="[API_ENDPOINT_URL]"
)


def run_llm(prompt, temperature=[SOME_TEMPERATURE_VALUE], max_tokens=[MAX_TOKENS_VALUE], model="[MODEL_NAME]"):
    system_message = "You are a helpful assistant that provides precise answers."
    messages = [{"role": "system", "content": system_message}, {"role": "user", "content": prompt}]
    for _ in range([NUM_RETRIES]):
        try:
            response = client.chat.completions.create(model=model, messages=messages, temperature=temperature,
                                                      max_tokens=max_tokens)
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM request failed: {e}, retrying in [RETRY_DELAY_IN_SECONDS] seconds...")
            time.sleep([RETRY_DELAY_IN_SECONDS])
    return ""


def get_legal_term_count(text: str) -> int:
    prompt = "[A prompt asking the LLM to count the number of legal terms in the following text and return only an integer.]"
    response_text = run_llm(prompt, max_tokens=20)
    match = re.search(r'\d+', response_text)
    if match:
        return int(match.group(0))
    print(f"Warning: Could not parse legal term count from '{response_text}', defaulting to 0.")
    return 0


def get_llm_complexity_score(text: str) -> float:
    prompt = "[A prompt asking the LLM to act as a legal expert and rate the complexity of the following text on a scale of 1-10, returning only a number.]"
    response_text = run_llm(prompt, max_tokens=20)
    match = re.search(r'\d+\.?\d*', response_text)
    if match:
        return float(match.group(0))
    print(f"Warning: Could not parse complexity score from '{response_text}', defaulting to 1.0.")
    return 1.0


class LegalComplexityAnalyzer:
    def __init__(self):
        self.gfi_mod_min = self.gfi_mod_max = None
        self.fkgl_min = self.fkgl_max = None

    def _calculate_raw_scores(self, text: str) -> Dict[str, Any]:
        word_count = textstat.lexicon_count(text)
        if word_count == 0:
            raw_gfi_mod = 0.0
        else:
            avg_sentence_length = textstat.avg_sentence_length(text)
            legal_term_count = get_legal_term_count(text)
            raw_gfi_mod = 0.4 * (avg_sentence_length + (legal_term_count * 100 / word_count))
        raw_fkgl = textstat.flesch_kincaid_grade(text)
        raw_llm_score = get_llm_complexity_score(text)
        return {"raw_gfi_mod": raw_gfi_mod, "raw_fkgl": raw_fkgl, "raw_llm_score": raw_llm_score}

    def calibrate(self, all_raw_scores: List[Dict[str, Any]]):
        gfi_scores = [item['raw_gfi_mod'] for item in all_raw_scores]
        fkgl_scores = [item['raw_fkgl'] for item in all_raw_scores]
        self.gfi_mod_min, self.gfi_mod_max = min(gfi_scores), max(gfi_scores)
        self.fkgl_min, self.fkgl_max = min(fkgl_scores), max(fkgl_scores)
        print("\n--- Calibration Complete ---")
        print(f"Modified GFI Range: Min={self.gfi_mod_min:.2f}, Max={self.gfi_mod_max:.2f}")
        print(f"FKGL Range: Min={self.fkgl_min:.2f}, Max={self.fkgl_max:.2f}")

    def get_normalized_scores(self, raw_scores: Dict[str, Any]) -> Dict[str, Any]:
        gfi_raw = raw_scores['raw_gfi_mod']
        fkgl_raw = raw_scores['raw_fkgl']
        norm_gfi = (gfi_raw - self.gfi_mod_min) / (self.gfi_mod_max - self.gfi_mod_min) if (
                                                                                                   self.gfi_mod_max - self.gfi_mod_min) != 0 else 0.5
        norm_fkgl = (fkgl_raw - self.fkgl_min) / (self.fkgl_max - self.fkgl_min) if (
                                                                                            self.fkgl_max - self.fkgl_min) != 0 else 0.5
        norm_llm = raw_scores['raw_llm_score'] / 10.0
        final_score = 0.4 * norm_gfi + 0.2 * norm_fkgl + 0.4 * norm_llm
        return {
            'norm_gfi_mod': round(norm_gfi, 3),
            'norm_fkgl': round(norm_fkgl, 3),
            'norm_llm_score': round(norm_llm, 3),
            'final_score': round(final_score, 3),
        }


def process_dataset(file_path: str, text_extraction_func: Callable[[Dict], Dict[str, str]]):
    print(f"\n===== Starting to process file: {os.path.basename(file_path)} =====")
    analyzer = LegalComplexityAnalyzer()

    print("--- Phase 1: Reading and Sampling Data ---")
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    random.shuffle(lines)
    sampling_ratio = 0.3  # Example sampling ratio
    subset_size = int(len(lines) * sampling_ratio)
    lines_subset = lines[:subset_size]

    print(
        f"Total lines in file: {len(lines)}, randomly processing {int(sampling_ratio * 100)}% of the data ({len(lines_subset)} lines).")

    print("\n--- Phase 2: Calculating Raw Scores for Sampled Data ---")
    all_items = []
    for i, line in enumerate(lines_subset):
        data = json.loads(line)
        extracted_texts = text_extraction_func(data)
        text_for_analysis = extracted_texts['text_for_analysis']
        classification_prompt = extracted_texts['classification_prompt']

        print(f"Processing... {i + 1}/{len(lines_subset)} (idx: {data.get('idx', 'N/A')})")
        raw_scores = analyzer._calculate_raw_scores(text_for_analysis)
        all_items.append({
            'idx': data.get('idx'),
            'text_for_analysis': text_for_analysis,
            'classification_prompt': classification_prompt,
            **raw_scores
        })

    analyzer.calibrate(all_items)

    print("\n--- Phase 3: Calculating Final Scores ---")
    results_list = []
    for item in all_items:
        normalized_scores = analyzer.get_normalized_scores(item)
        results_list.append({
            'idx': item['idx'],
            'text_for_analysis': item['text_for_analysis'],
            'classification_prompt': item['classification_prompt'],
            'raw_gfi_mod': round(item['raw_gfi_mod'], 2),
            'raw_fkgl': round(item['raw_fkgl'], 2),
            'raw_llm_score': item['raw_llm_score'],
            **normalized_scores
        })
    df = pd.DataFrame(results_list)

    print("\n--- Phase 4: Dynamic Classification using Quantiles ---")
    try:
        df['classification'] = pd.qcut(df['final_score'], q=3, labels=["1", "2", "3"], duplicates='drop')
    except ValueError:
        print("Warning: Cannot divide data into 3 unique bins. Using fixed thresholds as a fallback.")
        df['classification'] = df['final_score'].apply(
            lambda x: "1" if x < 1 / 3 else ("2" if x < 2 / 3 else "3")
        )

    print("\n--- Classification Results Statistics ---")
    print(df['classification'].value_counts().sort_index())

    df = df.rename(columns={
        'classification': 'label',
        'classification_prompt': 'question_text'
    })

    output_filename = os.path.basename(file_path).replace('.jsonl', '_complexity.csv')
    output_path = os.path.join(evaluate_folder, output_filename)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ Processing complete! Results saved to: {output_path}")


def extract_texts_for_bar_exam(data: Dict) -> Dict[str, str]:
    prompt_background = data.get("prompt", "")
    question = data.get("question", "")
    A, B, C, D = data["choice_a"], data["choice_b"], data["choice_c"], data["choice_d"]

    if prompt_background and str(prompt_background).strip().lower() != "nan":
        text_for_analysis = f"Background: {prompt_background}\n\nQuestion: {question}"
    else:
        text_for_analysis = f"Question: {question}"

    classification_prompt = (
        "Determine the complexity of the following legal multiple-choice question (simple, medium, or complex):\n"
        f"{text_for_analysis}\n\n"
        f"Choices:\nA: {A}\nB: {B}\nC: {C}\nD: {D}\n"
    )
    return {
        "text_for_analysis": text_for_analysis,
        "classification_prompt": classification_prompt
    }


def extract_texts_for_housing(data: Dict) -> Dict[str, str]:
    state = data.get("state", "").strip()
    question = data.get("question", "")
    text_for_analysis = f"State: {state}\n\nQuestion: {question}"
    classification_prompt = (
        "Classify the complexity level of the following legal Yes/No question as one of the following: Simple, Medium, or Complex.:\n"
        f"Question: {text_for_analysis}\n"
        "Answer:"
    )
    return {
        "text_for_analysis": text_for_analysis,
        "classification_prompt": classification_prompt
    }


if __name__ == "__main__":
    bar_exam_file = os.path.join(data_folder, '[BAR_EXAM_FILENAME].jsonl')
    if os.path.exists(bar_exam_file):
        process_dataset(bar_exam_file, extract_texts_for_bar_exam)

    housing_aux_file = os.path.join(data_folder, '[HOUSING_DATA_FILENAME].jsonl')
    if os.path.exists(housing_aux_file):
        process_dataset(housing_aux_file, extract_texts_for_housing)