# src/build_corpus.py

"""
build_corpus.py

Purpose:
  Reads the human-editable master corpus from a TSV file and converts it into
  a syntactically perfect, machine-readable JSON Lines (JSONL) file. This script
  is the first step in the data pipeline, ensuring the source data is clean
  and correctly formatted for all subsequent processing.

Inputs:
  - ../data/master_corpus.tsv: A Tab-Separated Values file containing the full
    question corpus with headers.

Outputs:
  - ../data/questions_en_uk.jsonl: The master English (UK) question corpus in
    JSONL format, with all strings and special characters correctly escaped.
"""

import csv
import json
from pathlib import Path

# --- Configuration ---
# Use pathlib to create robust, cross-platform paths relative to this script's location.
# Path(__file__).resolve() -> gets the full path to this script
# .parent -> gets the 'src' directory
# .parent -> gets the root project directory (Multi-Ling-Rubric/)
BASE_DIR = Path(__file__).resolve().parent.parent

SOURCE_TSV_FILE = BASE_DIR / 'data' / 'master_corpus.tsv'
OUTPUT_JSONL_FILE = BASE_DIR / 'data' / 'questions_en_uk.jsonl'

def main():
    """
    Main function to read the TSV and generate the JSONL file.
    """
    corpus = []
    print(f"Reading master corpus from: {SOURCE_TSV_FILE}")
    try:
        with open(SOURCE_TSV_FILE, 'r', encoding='utf-8') as f:
            # Use csv.DictReader with 'excel-tab' dialect for TSV files.
            reader = csv.DictReader(f, dialect='excel-tab')
            for row in reader:
                # If the reasoning field is empty in the TSV, it should be None.
                if not row.get('gold_standard_reasoning'):
                    row['gold_standard_reasoning'] = None
                corpus.append(row)
                
    except FileNotFoundError:
        print(f"Error: Master corpus file not found at '{SOURCE_TSV_FILE}'.")
        return
    except Exception as e:
        print(f"An error occurred while reading the TSV file: {e}")
        return

    # Ensure the output directory exists
    OUTPUT_JSONL_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Write the data to the JSONL file
    print(f"Generating JSONL file at: {OUTPUT_JSONL_FILE}")
    try:
        with open(OUTPUT_JSONL_FILE, 'w', encoding='utf-8') as f:
            for entry in corpus:
                # Exclude keys with None values for a cleaner output file.
                clean_entry = {k: v for k, v in entry.items() if v is not None}
                
                # json.dumps handles all necessary escaping automatically and correctly.
                json_line = json.dumps(clean_entry)
                f.write(json_line + '\n')
    except Exception as e:
        print(f"An error occurred while writing the JSONL file: {e}")
        return

    print(f"Successfully generated '{OUTPUT_JSONL_FILE.name}' with {len(corpus)} questions.")

if __name__ == '__main__':
    main()
