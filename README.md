# A Cross-Lingual Study in LLM Factual Accuracy and Procedural Reasoning

This repository contains the complete code, data, and methodology for a research study into the cross-lingual performance consistency of Large Language Models (LLMs). The primary goal of this work is to quantify "performance drift" in a model's foundational capabilities when tested in different languages.

Our findings demonstrate that high performance on English-centric benchmarks is not a reliable indicator of a model's performance in other languages, with procedural reasoning being particularly susceptible to degradation. This work proposes a robust, automated methodology for conducting this essential due diligence.

---

## Table of Contents

- [Abstract](#abstract)
- [Repository Structure](#repository-structure)
- [Methodology Overview](#methodology-overview)
- [Installation and Setup](#installation-and-setup)
- [How to Replicate the Study](#how-to-replicate-the-study)
- [Citing This Work](#citing-this-work)

---

## Abstract

The global deployment of Large Language Models necessitates robust, evidence-based assurance of their performance across diverse linguistic contexts. This study investigates a critical and under-examined issue: the consistency of a model's foundational competence in Factual Accuracy and Procedural Reasoning when tested in UK English (EN), German (DE), and Spanish (ES). We introduce a novel, fully automated pipeline, including a "Round-Trip Translation" quality gate to ensure translational equivalence and a "Hybrid Automated Scoring" protocol to objectively evaluate model reasoning. Our results provide empirical evidence of significant "performance drift," showing that a model's reliability can degrade substantially in non-English languages. We find that procedural reasoning is significantly more brittle to linguistic shifts than factual recall. These findings carry a stark warning for any multicultural nation or global organisation: relying on monolingual benchmarks is insufficient and can mask critical performance deficits, leading to the deployment of unreliable and potentially inequitable AI systems.

---

## Repository Structure
Multi-Ling-Rubric/
│
├── .gitignore
├── README.md
│
├── requirements.txt # Python dependencies
│
├── src/ # All Python source code
│ ├── build_corpus.py # Generates the final JSONL corpus from the master TSV
│ ├── translate_corpus.py # Creates and validates multilingual question files
│ ├── run_experiments.py # Executes the tests against the Ollama API
│ ├── score_results.py # Applies the hybrid automated scoring logic
│ └── analyse_results.py # Generates all final tables and visualizations
│
├── data/ # Core data assets
│ └── master_corpus.tsv # The human-editable master list of 350 questions
│
└── paper/ # Final research paper
└── cross_lingual_study.pdf


---

## Methodology Overview

The experimental process is executed via a sequence of scripts in the `src/` directory.

1.  **Corpus Generation:** `build_corpus.py` reads the human-friendly `data/master_corpus.tsv` and generates a syntactically perfect `questions_en_uk.jsonl`.
2.  **Translation:** `translate_corpus.py` takes the English corpus, translates it to DE and ES, and applies a "Round-Trip Translation" quality gate to ensure high semantic fidelity.
3.  **Execution:** `run_experiments.py` systematically queries a list of LLMs (via Ollama) for every validated question in every language and saves the raw JSON responses.
4.  **Scoring:** `score_results.py` applies a hybrid automated scoring rubric, using regex for final answers and semantic similarity to judge the quality of reasoning against a gold standard.
5.  **Analysis:** `analyse_results.py` reads the final scored data to produce all the summary tables and visualizations used in the paper.

---

## Installation and Setup

This project uses Python 3.8+ and `venv` for dependency management.

1.  **Clone the repository:**
    ```bash
    git clone [URL_to_your_repo]
    cd Multi-Ling-Rubric
    ```

2.  **Create and activate a Python virtual environment:**
    ```bash
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: The first time you run a script that uses the sentence-transformer, it will automatically download the required pre-trained model.*

---

## How to Replicate the Study

To run the entire pipeline from start to finish, execute the scripts in the `src/` directory in the following order. **Ensure your Ollama server is running before starting Step 3.**

1.  **Generate the JSONL corpus:**
    ```bash
    python src/build_corpus.py
    ```

2.  **Create the multilingual question files:**
    ```bash
    python src/translate_corpus.py
    ```

3.  **Run the experiments against the LLMs:**
    ```bash
    python src/run_experiments.py
    ```
    *This is a time-consuming step.*

4.  **Score the raw results:**
    ```bash
    python src/score_results.py
    ```

5.  **Generate the final analysis and plots:**
    ```bash
    python src/analyse_results.py
    ```
    All final CSV tables and PNG charts will be saved in the `analysis_outputs/` directory.

---

## Citing This Work

If you use this methodology or data in your research, please cite it as follows:
Hejroe. (2025). A Cross-Lingual Study in LLM Factual Accuracy and Procedural Reasoning. GitHub Repository. https://github.com/hejroe/Multi-Ling-Rubric
