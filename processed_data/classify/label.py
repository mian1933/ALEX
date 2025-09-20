import os
import time
import json
import pandas as pd
import textstat
import re
from openai import OpenAI
from typing import List, Dict, Any, Callable
from tqdm import tqdm

# --- Configuration ---
LABELED_DATA_FOLDER = "[PATH_TO_YOUR_LABELED_DATA]"
UNLABELED_DATA_FOLDER = "[PATH_TO_YOUR_UNLABELED_DATA]"
EVALUATION_FOLDER = "[PATH_TO_YOUR_COMPLEXITY_ANALYSIS_OUTPUT]"
os.makedirs(EVALUATION_FOLDER, exist_ok=True)

client = OpenAI(
    api_key="[YOUR_API_KEY]",
    base_url="[API_ENDPOINT_URL]"
)


# --- LLM and Scoring Functions ---

def run_llm(prompt, temperature=0.1, max_tokens=256, model="[MODEL_NAME]"):
    system_message = "You are a helpful assistant that provides precise answers."
    messages = [{"role": "system", "content": system_message}, {"role": "user", "content": prompt}]
    for _ in range(3):
        try:
            response = client.chat.completions.create(model=model, messages=messages, temperature=temperature,
                                                      max_tokens=max_tokens)
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM request failed: {e}, retrying in 2 seconds...")
            time.sleep(2)
    return ""


def get_legal_term_count(text: str) -> int:
    prompt = "[A prompt asking the LLM to count legal terms and return only an integer.]"
    response_text = run_llm(prompt, max_tokens=20)
    match = re.search(r'\d+', response_text)
    if match:
        return int(match.group(0))
    print(f"Warning: Could not parse legal term count from '{response_text}', defaulting to 0.")
    return 0


def get_llm_complexity_score(text: str) -> float:
    prompt = "[A prompt asking the LLM to rate text complexity on a 1-10 scale and return only a number.]"
    response_text = run_llm(prompt, max_tokens=20)
    match = re.search(r'\d+\.?\d*', response_text)
    if match:
        return float(match.group(0))
    print(f"Warning: Could not parse complexity score from '{response_text}', defaulting to 1.0.")
    return 1.0


# --- Core Analyzer Class (for Labeled Data) ---

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

    def get_final_analysis(self, raw_scores: Dict[str, Any]) -> Dict[str, Any]:
        gfi_raw = raw_scores['raw_gfi_mod']
        fkgl_raw = raw_scores['raw_fkgl']
        norm_gfi = (gfi_raw - self.gfi_mod_min) / (self.gfi_mod_max - self.gfi_mod_min) if (
                                                                                                       self.gfi_mod_max - self.gfi_mod_min) != 0 else 0.5
        norm_fkgl = (fkgl_raw - self.fkgl_min) / (self.fkgl_max - self.fkgl_min) if (
                                                                                                self.fkgl_max - self.fkgl_min) != 0 else 0.5
        norm_llm = raw_scores['raw_llm_score'] / 10.0
        final_score = 0.4 * norm_gfi + 0.2 * norm_fkgl + 0.4 * norm_llm
        return {'final_score': round(final_score, 3)}


# --- Text Extraction Functions ---

def extract_texts_for_bar_exam(data: Dict) -> Dict[str, str]:
    prompt_background = data.get("prompt", "")
    question = data.get("question", "")
    A, B, C, D = data["choice_a"], data["choice_b"], data["choice_c"], data["choice_d"]
    if prompt_background and str(prompt_background).strip().lower() != "nan":
        text_for_analysis = f"Background: {prompt_background}\n\nQuestion: {question}"
    else:
        text_for_analysis = f"Question: {question}"
    classification_prompt = (
        f"Determine the complexity of the following legal multiple-choice question (simple, medium, or complex):\n"
        f"{text_for_analysis}\n\nChoices:\nA: {A}\nB: {B}\nC: {C}\nD: {D}\n"
    )
    return {"text_for_analysis": text_for_analysis, "classification_prompt": classification_prompt}


def extract_texts_for_housing(data: Dict) -> Dict[str, str]:
    state = data.get("state", "").strip()
    question = data.get("question", "")
    text_for_analysis = f"State: {state}\n\nQuestion: {question}"
    classification_prompt = (
        f"Classify the complexity level of the following legal Yes/No question as one of the following: Simple, Medium, or Complex.:\n"
        f"Question: {text_for_analysis}\nAnswer:"
    )
    return {"text_for_analysis": text_for_analysis, "classification_prompt": classification_prompt}


# --- Dataset Processors ---

def process_labeled_dataset(file_path: str, text_extraction_func: Callable[[Dict], Dict[str, str]]):
    print(f"\n===== Starting to process LABELED file: {os.path.basename(file_path)} =====")
    analyzer = LegalComplexityAnalyzer()

    print("--- Phase 1: Calculating raw scores for all samples ---")
    all_items = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i, line in tqdm(enumerate(lines), total=len(lines), desc=f"Analyzing {os.path.basename(file_path)}"):
            data = json.loads(line)
            extracted_texts = text_extraction_func(data)
            raw_scores = analyzer._calculate_raw_scores(extracted_texts['text_for_analysis'])
            all_items.append({'idx': data.get('idx'), 'classification_prompt': extracted_texts['classification_prompt'],
                              **raw_scores})

    analyzer.calibrate(all_items)

    print("\n--- Phase 2: Calculating final scores and classifying ---")
    results_list = []
    for item in all_items:
        final_analysis = analyzer.get_final_analysis(item)
        results_list.append({'idx': item['idx'], 'classification_prompt': item['classification_prompt'],
                             'raw_llm_score': item['raw_llm_score'], **final_analysis})

    df = pd.DataFrame(results_list)
    try:
        df['label'] = pd.qcut(df['final_score'], q=3, labels=["1", "2", "3"], duplicates='drop')
    except ValueError:
        print("Warning: Cannot divide data into 3 unique bins. Using fixed thresholds as a fallback.")
        df['label'] = df['final_score'].apply(lambda x: "1" if x < 1 / 3 else ("2" if x < 2 / 3 else "3"))

    print("\n--- Classification Results Statistics ---")
    print(df['label'].value_counts().sort_index())

    output_filename = os.path.basename(file_path).replace('.jsonl', '_complexity.csv')
    output_path = os.path.join(EVALUATION_FOLDER, output_filename)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ Processing complete! Labeled data results saved to: {output_path}")


def process_unlabeled_dataset(file_path: str, text_extraction_func: Callable[[Dict], Dict[str, str]]):
    print(f"\n===== Starting to process UNLABELED file: {os.path.basename(file_path)} =====")
    determine_prompts = []
    classify_prompts = []

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i, line in tqdm(enumerate(lines), total=len(lines),
                            desc=f"Generating prompts for {os.path.basename(file_path)}"):
            data = json.loads(line)
            extracted_texts = text_extraction_func(data)
            idx = data.get('idx', f'unlabeled_{i}')
            determine_prompts.append({'idx': idx, 'question_text': extracted_texts['text_for_analysis']})
            classify_prompts.append({'idx': idx, 'question_text': extracted_texts['classification_prompt']})

    df_determine = pd.DataFrame(determine_prompts)
    determine_output_filename = os.path.basename(file_path).replace('.jsonl', '_determine_prompts.csv')
    determine_output_path = os.path.join(EVALUATION_FOLDER, determine_output_filename)
    df_determine.to_csv(determine_output_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 'Determine' prompts saved to: {determine_output_path}")

    df_classify = pd.DataFrame(classify_prompts)
    classify_output_filename = os.path.basename(file_path).replace('.jsonl', '_classify_prompts.csv')
    classify_output_path = os.path.join(EVALUATION_FOLDER, classify_output_filename)
    df_classify.to_csv(classify_output_path, index=False, encoding='utf-8-sig')
    print(f"✅ 'Classify' prompts saved to: {classify_output_path}")


# --- Main Execution ---
if __name__ == "__main__":
    # Process Labeled Datasets for complexity analysis
    print("\n--- Processing LABELED Datasets for Complexity Analysis ---")
    labeled_bar_exam_file = os.path.join(LABELED_DATA_FOLDER, '[LABELED_BAR_EXAM_FILENAME].jsonl')
    if os.path.exists(labeled_bar_exam_file):
        process_labeled_dataset(labeled_bar_exam_file, extract_texts_for_bar_exam)

    labeled_housing_file = os.path.join(LABELED_DATA_FOLDER, '[LABELED_HOUSING_FILENAME].jsonl')
    if os.path.exists(labeled_housing_file):
        process_labeled_dataset(labeled_housing_file, extract_texts_for_housing)

    # Process Unlabeled Datasets to generate prompts
    print("\n--- Processing UNLABELED Datasets to Generate Prompts ---")
    unlabeled_bar_exam_file = os.path.join(UNLABELED_DATA_FOLDER, '[UNLABELED_BAR_EXAM_FILENAME].jsonl')
    if os.path.exists(unlabeled_bar_exam_file):
        process_unlabeled_dataset(unlabeled_bar_exam_file, extract_texts_for_bar_exam)

    unlabeled_housing_file = os.path.join(UNLABELED_DATA_FOLDER, '[UNLABELED_HOUSING_FILENAME].jsonl')
    if os.path.exists(unlabeled_housing_file):
        process_unlabeled_dataset(unlabeled_housing_file, extract_texts_for_housing)

    print("\nAll tasks completed.")