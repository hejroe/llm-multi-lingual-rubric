# src/run_experiments.py

"""
run_experiments.py

Purpose:
  Executes the core experimental runs. It systematically queries a list of
  LLMs for every validated question in every language (EN, DE, ES) and saves
  the complete, raw JSON responses to a timestamped results file.

Inputs:
  - ../data/questions_en_uk.jsonl
  - ../translation_outputs/questions_de_[latest].jsonl
  - ../translation_outputs/questions_es_[latest].jsonl

Outputs:
  - ../experimental_results/raw_results_[timestamp].jsonl: A single, large
    JSONL file containing all raw model responses for the entire experiment.
"""

import json
import requests
from datetime import datetime, timezone
from tqdm import tqdm
import time
import os
import glob
from pathlib import Path

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent
OLLAMA_API_ENDPOINT = 'http://localhost:11434/api/generate'
MODELS_TO_TEST = [
    "llama3:8b", "llama3.1:8b", "llama3.2:3b", "falcon3:10b", "gpt-oss:20b",
    "deepseek-r1:8b", "qwen3:8b", "phi4:14b", "granite3.3:8b",
    "gemma3:12b", "gemma3n:e4b"
]

# --- Dynamic File Naming and Finding ---
TIMESTAMP_UTC = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
OUTPUT_DIR = BASE_DIR / 'experimental_results'
RAW_RESULTS_FILE = OUTPUT_DIR / f'raw_results_{TIMESTAMP_UTC}.jsonl'

def find_latest_file(pattern):
    """Finds the most recently created file matching a pattern."""
    try:
        list_of_files = glob.glob(pattern)
        if not list_of_files: return None
        return max(list_of_files, key=os.path.getctime)
    except Exception as e:
        print(f"Error finding file for pattern {pattern}: {e}")
        return None

# Use Path objects for robust file path construction
QUESTION_FILES = {
    'EN': BASE_DIR / 'data' / 'questions_en_uk.jsonl',
    'DE': find_latest_file(str(BASE_DIR / 'translation_outputs' / 'questions_de_*.jsonl')),
    'ES': find_latest_file(str(BASE_DIR / 'translation_outputs' / 'questions_es_*.jsonl'))
}

def query_model(model_name, prompt):
    """Sends a prompt to a model via the Ollama API."""
    payload = {"model": model_name, "prompt": prompt, "stream": False}
    try:
        response = requests.post(OLLAMA_API_ENDPOINT, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

def main():
    """Main function to run all experiments."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("--- Starting Experiment Run ---")
    print(f"Results will be saved to: {RAW_RESULTS_FILE.name}")
    
    test_counter = 0
    with open(RAW_RESULTS_FILE, 'w') as f: pass 

    for lang_code, file_path in QUESTION_FILES.items():
        if file_path is None or not os.path.exists(file_path):
            print(f"Warning: No question file found for language {lang_code}. Skipping.")
            continue
            
        print(f"\nLoading questions for language: {lang_code} from '{os.path.basename(file_path)}'")
        with open(file_path, 'r', encoding='utf-8') as f:
            questions = [json.loads(line) for line in f]

        total_tests_for_lang = len(MODELS_TO_TEST) * len(questions)
        
        with tqdm(total=total_tests_for_lang, desc=f"Testing Language: {lang_code}") as pbar:
            for model in MODELS_TO_TEST:
                for question in questions:
                    prompt = question.get('question_text', question.get('question_text_english'))
                    if not prompt:
                        pbar.update(1)
                        continue

                    raw_response = query_model(model, prompt)
                    
                    result_record = {
                        "test_id": f"run_{TIMESTAMP_UTC}_{test_counter:05d}",
                        "question_id": question['question_id'],
                        "model_identifier": model,
                        "language": lang_code,
                        "prompt_text": prompt,
                        "raw_response": raw_response,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat()
                    }

                    with open(RAW_RESULTS_FILE, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(result_record) + '\n')
                    
                    test_counter += 1
                    pbar.update(1)

    print(f"\n--- Experiment run complete. ---")
    print(f"Saved {test_counter} results to '{RAW_RESULTS_FILE}'.")

if __name__ == '__main__':
    main()
