import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Dict

# --- Configuration ---
# The specific path to the trained model is replaced with a placeholder.
MODEL_PATH = "[PATH_TO_YOUR_TRAINED_COMPLEXITY_MODEL]"


class ComplexityModel:
    """
    A class that encapsulates the logic for model loading, prompt creation, and prediction.
    """

    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the model and tokenizer
        print(f"Loading model from '{model_path}'...")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("✅ Model and tokenizer loaded successfully.")

    def _create_mcq_prompt(self, background: str, question: str, choices: Dict[str, str]) -> str:
        """
        Creates the prompt for multiple-choice questions.
        The specific wording is abstracted to a template format.
        """
        # This function should format the inputs into a prompt that asks the model
        # to classify the complexity of the given multiple-choice question.
        full_context = f"Background: {background}\n\nQuestion: {question}" if background else f"Question: {question}"
        choices_text = f"A: {choices.get('A', '')}\nB: {choices.get('B', '')}..."  # etc.

        prompt = f"[Your prompt to determine MCQ complexity, including context and choices]\n{full_context}\n{choices_text}\nAnswer:"
        return prompt

    def _create_yesno_prompt(self, state: str, question: str) -> str:
        """
        Creates the prompt for Yes/No questions.
        The specific wording is abstracted to a template format.
        """
        # This function should format the inputs into a prompt that asks the model
        # to classify the complexity of the given Yes/No question.
        full_context = f"State: {state}\n\nQuestion: {question}"

        prompt = f"[Your prompt to classify Yes/No question complexity]\n{full_context}\nAnswer:"
        return prompt

    def predict_complexity(self, prompt: str) -> str:
        """
        Takes a formatted prompt and performs model prediction.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        output_ids = self.model.generate(**inputs, max_new_tokens=10)
        pred_str = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip().lower()

        if pred_str in ["simple", "medium", "complex"]:
            return pred_str
        else:
            print(f"Warning: Model produced a non-standard output: '{pred_str}'")
            return "unknown"


# --- Global Singleton: Load the model once when the module is first imported ---
try:
    _analyzer_instance = ComplexityModel(model_path=MODEL_PATH)
except Exception as e:
    print(f"Fatal Error: Failed to initialize the complexity analysis model. Error: {e}")
    _analyzer_instance = None


# --- Public-facing Functions ---

def classify_multiple_choice_complexity(background: str, question: str, choice_a: str, choice_b: str, choice_c: str,
                                        choice_d: str) -> str:
    """
    Analyzes the complexity of a multiple-choice question.

    Pass the different parts of the question, and the function will handle
    prompt creation and model prediction internally.

    Returns:
        str: "simple", "medium", "complex", or "unknown"
    """
    if _analyzer_instance is None:
        return "error: model not initialized"

    choices = {"A": choice_a, "B": choice_b, "C": choice_c, "D": choice_d}
    prompt = _analyzer_instance._create_mcq_prompt(background, question, choices)
    return _analyzer_instance.predict_complexity(prompt)


def classify_yes_no_complexity(state: str, question: str) -> str:
    """
    Analyzes the complexity of a Yes/No question.

    Pass the different parts of the question, and the function will handle
    prompt creation and model prediction internally.

    Returns:
        str: "simple", "medium", "complex", or "unknown"
    """
    if _analyzer_instance is None:
        return "error: model not initialized"

    prompt = _analyzer_instance._create_yesno_prompt(state, question)
    return _analyzer_instance.predict_complexity(prompt)