# src/score_results.py

"""
score_results.py

Purpose:
  Applies the final, calibrated Hybrid Automated Scoring protocol to the raw
  experimental results. This is a fully automated script that categorises every
  response, removing the need for subjective manual review.

Inputs:
  - ../experimental_results/raw_results_[latest].jsonl: The raw experimental data.
  - ../data/questions_en_uk.jsonl: The master corpus with answers and reasoning.

Outputs:
  - ../analysis_outputs/final_scored_results_[timestamp].csv: The final, clean,
    and fully scored dataset ready for analysis.
"""

import json
import pandas as pd
import re
from tqdm import tqdm
import os
import glob
from sentence_transformers import SentenceTransformer, util
from datetime import datetime, timezone
from pathlib import Path

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_RESULTS_DIR = BASE_DIR / 'experimental_results'
ANALYSIS_DIR = BASE_DIR / 'analysis_outputs'
MASTER_CORPUS_FILE = BASE_DIR / 'data' / 'questions_en_uk.jsonl'

# --- Scoring Logic Constants ---
SCORE_CORRECT = 1.0
SCORE_CORRECT_PROCESS = 0.5
SCORE_IDK = 0.25
SCORE_AMBIGUOUS = 0.0
SCORE_INCORRECT_GUESS = -0.5
SCORE_FABRICATION = -1.0
SCORE_INCORRECT = -1.0

# --- FINAL CALIBRATED THRESHOLDS ---
REASONING_SIMILARITY_HIGH = 0.70
REASONING_SIMILARITY_LOW = 0.60

IDK_KEYWORDS = ["i don't know", "i do not know", "cannot answer", "unable to answer", "as an ai", "i am unable"]

# --- Load Semantic Similarity Model ---
print("Loading semantic similarity model...")
try:
    similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading SentenceTransformer model: {e}")
    exit()


def find_latest_file(directory, pattern):
    """Finds the most recently created file in a directory matching a pattern."""
    try:
        search_path = os.path.join(directory, pattern)
        list_of_files = glob.glob(search_path)
        if not list_of_files: return None
        return max(list_of_files, key=os.path.getctime)
    except Exception as e:
        print(f"Error finding file for pattern {pattern}: {e}")
        return None

def load_all_questions():
    """Loads the single, master EN(UK) question file."""
    all_questions = {}
    if not MASTER_CORPUS_FILE.exists():
        print(f"Error: Master question file '{MASTER_CORPUS_FILE}' not found.")
        return None
    
    with open(MASTER_CORPUS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                question = json.loads(line)
                all_questions[question['question_id']] = question
            except json.JSONDecodeError:
                print(f"Warning: Skipping corrupted line in {MASTER_CORPUS_FILE}")
    return all_questions

def score_response(row, questions_dict):
    """
    Applies the hybrid automated scoring logic to a single result row.
    Returns a tuple of (score, category, reasoning_similarity).
    """
    question_id = row['question_id']
    question_data = questions_dict.get(question_id)
    
    if not question_data: return 0, "QuestionDataMissing", None

    raw_response_data = row.get('raw_response', {})
    if not isinstance(raw_response_data, dict): return 0, "MalformedResponse", None
        
    response_text = raw_response_data.get('response', '').lower()
    
    if not response_text or "error" in raw_response_data: return 0, "APIError", None

    # 1. Check for IDK first
    if any(keyword in response_text for keyword in IDK_KEYWORDS):
        return SCORE_IDK, "IDK", None

    # 2. Check for correct final answer
    regex = question_data.get('answer_format_regex', '^$')
    is_answer_correct = bool(re.search(regex.lower(), response_text))
    
    # --- Apply Scoring Decision Tree ---
    # Factual Accuracy questions have simple scoring
    if question_data.get('domain') == "Factual Accuracy":
        score = SCORE_CORRECT if is_answer_correct else SCORE_INCORRECT
        category = "Correct" if is_answer_correct else "Incorrect"
        return score, category, None

    # Procedural Reasoning questions use the hybrid logic
    elif question_data.get('domain') == "Procedural Reasoning":
        gold_reasoning = question_data.get('gold_standard_reasoning')
        
        if not gold_reasoning:
            # If a reasoning question has no gold standard, we can't judge it. Score as ambiguous.
            return SCORE_AMBIGUOUS, "MissingGoldReasoning", None

        # Calculate reasoning similarity
        try:
            embedding1 = similarity_model.encode(gold_reasoning, convert_to_tensor=True)
            embedding2 = similarity_model.encode(response_text, convert_to_tensor=True)
            reasoning_score = util.pytorch_cos_sim(embedding1, embedding2).item()
        except Exception:
            return SCORE_AMBIGUOUS, "SimilarityError", None

        # Apply the final, fully automated logic
        if is_answer_correct:
            if reasoning_score >= REASONING_SIMILARITY_HIGH:
                return SCORE_CORRECT, "Correct", reasoning_score
            elif reasoning_score < REASONING_SIMILARITY_LOW:
                return SCORE_FABRICATION, "Fabrication", reasoning_score
            else: # Ambiguous reasoning is a final neutral category
                return SCORE_AMBIGUOUS, "AmbiguousReasoning", reasoning_score
        else: # Answer is incorrect
            if reasoning_score >= REASONING_SIMILARITY_HIGH:
                return SCORE_CORRECT_PROCESS, "CorrectProcess_IncorrectResult", reasoning_score
            elif reasoning_score < REASONING_SIMILARITY_LOW:
                return SCORE_INCORRECT, "Incorrect", reasoning_score
            else: # Ambiguous reasoning is a final neutral category
                return SCORE_AMBIGUOUS, "AmbiguousReasoning", reasoning_score

    # Fallback for any other case
    return SCORE_INCORRECT, "UnknownDomain_Incorrect", None

def robust_read_jsonl(file_path):
    """Reads a JSON Lines file, skipping any corrupted lines."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if not line.strip(): continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Warning: Skipping corrupted JSON object on line {i+1} in {file_path}")
    return pd.DataFrame(data)

def main():
    """Main function to orchestrate the scoring process."""
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    
    raw_results_file = find_latest_file(RAW_RESULTS_DIR, 'raw_results_*.jsonl')
    if not raw_results_file:
        print(f"Error: No raw results file found in '{RAW_RESULTS_DIR}'. Please run run_experiments.py first.")
        return

    print("Loading questions...")
    questions_dict = load_all_questions()
    if not questions_dict: return
    
    print(f"Loading results from '{os.path.basename(raw_results_file)}'...")
    results_df = robust_read_jsonl(raw_results_file)
    if results_df.empty:
        print("Error: The results file is empty or could not be read.")
        return

    print("Applying final, fully automated scoring logic...")
    tqdm.pandas(desc="Scoring responses")
    
    scored_data = results_df.progress_apply(lambda row: score_response(row, questions_dict), axis=1)
    results_df[['score', 'score_category', 'reasoning_similarity']] = pd.DataFrame(scored_data.tolist(), index=results_df.index)
    
    results_df['domain'] = results_df['question_id'].apply(lambda qid: questions_dict.get(qid, {}).get('domain'))
    
    output_columns = ['question_id', 'model_identifier', 'language', 'domain', 'score', 'score_category', 'reasoning_similarity', 'prompt_text']
    final_df = results_df[output_columns]
    
    timestamp_utc = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    scored_results_file = ANALYSIS_DIR / f'final_scored_results_{timestamp_utc}.csv'
    
    final_df.to_csv(scored_results_file, index=False, encoding='utf-8-sig')

    ambiguous_count = len(final_df[final_df['score_category'] == 'AmbiguousReasoning'])
    print(f"\n--- Final scoring complete. ---")
    print(f"Saved final scored results to '{scored_results_file}'.")
    print(f"A total of {ambiguous_count} responses were categorised as 'AmbiguousReasoning' with a neutral score of 0.0.")
    print("The scoring pipeline is now 100% automated. The dataset is complete.")

if __name__ == '__main__':
    main()
