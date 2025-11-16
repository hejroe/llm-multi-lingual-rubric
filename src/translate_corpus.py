"""
translate_corpus.py

Purpose:
  Takes the master English JSONL corpus and translates the questions into the
  target languages (DE, ES). It applies a 'Round-Trip Translation' quality
  gate, using semantic similarity to ensure a high degree of translational
  fidelity.

Authors:
  Hejroe, Gemini

Version:
  1.0

Last Updated:
  16 November 2025

Pre-requisites:
  - An internet connect is required for the translation.

Inputs:
  - ../data/questions_en_uk.jsonl: The master English (UK) corpus.

Outputs:
  - ../translation_outputs/questions_[lang]_[timestamp].jsonl: Validated,
    translated question files for each target language.
  - ../translation_outputs/translation_log_[timestamp].txt: A simple log file
    detailing the Pass/Fail status for each question's translation.

License:
  MIT License
----------------------------------------------------------------------------------
"""

import json
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from datetime import datetime, timezone
from pathlib import Path

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent
SOURCE_FILE = BASE_DIR / 'data' / 'questions_en_uk.jsonl'
OUTPUT_DIR = BASE_DIR / 'translation_outputs'
TARGET_LANGUAGES = ['de', 'es']
SIMILARITY_THRESHOLD = 0.95

# --- File Naming with Timestamp ---
TIMESTAMP_UTC = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
LOG_FILE = OUTPUT_DIR / f'translation_log_{TIMESTAMP_UTC}.txt'

# --- Load Models ---
print("Loading semantic similarity model (this may take a moment on first run)...")
try:
    similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading SentenceTransformer model. Ensure you have an internet connection.")
    print(f"Error: {e}")
    exit()

def translate_and_validate(question_text, target_lang):
    """
    Performs a round-trip translation and validates semantic similarity.
    Returns (translated_text, score, status).
    """
    try:
        to_target_translator = GoogleTranslator(source='en', target=target_lang)
        translated_text = to_target_translator.translate(question_text)

        from_target_translator = GoogleTranslator(source=target_lang, target='en')
        round_trip_text = from_target_translator.translate(translated_text)

        embedding1 = similarity_model.encode(question_text, convert_to_tensor=True)
        embedding2 = similarity_model.encode(round_trip_text, convert_to_tensor=True)
        cosine_score = util.pytorch_cos_sim(embedding1, embedding2).item()

        if cosine_score >= SIMILARITY_THRESHOLD:
            return translated_text, cosine_score, "Pass"
        else:
            return None, cosine_score, "Fail"
            
    except Exception as e:
        return None, 0, f"Error: {e}"

def main():
    """
    Main function to process the source file and generate translated files and a log.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
            source_questions = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"Error: Source file '{SOURCE_FILE}' not found. Please run build_corpus.py first.")
        return

    # Prepare the log file header
    with open(LOG_FILE, 'w', encoding='utf-8') as log_f:
        log_f.write(f"Translation Validation Log - {TIMESTAMP_UTC}\n")
        log_f.write(f"Source File: {SOURCE_FILE.name}\n")
        log_f.write(f"Similarity Threshold: {SIMILARITY_THRESHOLD}\n")
        log_f.write("="*50 + "\n")

    for lang in TARGET_LANGUAGES:
        print(f"\n--- Processing translations for language: {lang.upper()} ---")
        translated_corpus = []
        
        with open(LOG_FILE, 'a', encoding='utf-8') as log_f:
            log_f.write(f"\n--- Language: {lang.upper()} ---\n")

            for question in tqdm(source_questions, desc=f"Translating to {lang.upper()}"):
                question_id = question['question_id']
                original_text = question['question_text_english']
                
                translated_text, score, status = translate_and_validate(original_text, lang)

                log_f.write(f"{question_id}: {status}\n")

                if status == "Pass":
                    new_question = question.copy()
                    new_question['question_text'] = translated_text
                    del new_question['question_text_english'] 
                    translated_corpus.append(new_question)

        # Write the validated corpus to its timestamped file
        output_file = OUTPUT_DIR / f"questions_{lang}_{TIMESTAMP_UTC}.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in translated_corpus:
                f.write(json.dumps(item) + '\n')
        
        pass_count = len(translated_corpus)
        total_count = len(source_questions)
        fail_count = total_count - pass_count
        
        print(f"Successfully created '{output_file.name}'.")
        print(f"Results for {lang.upper()}: {pass_count} Passed / {fail_count} Failed.")
    
    print(f"\nTranslation process complete. Log saved to '{LOG_FILE}'.")

if __name__ == '__main__':
    main()
