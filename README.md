# The Leaderboard Illusion: A Risk Analysis of Performance Degradation in Multilingual Large Language Models

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

The selection of Large Language Models (LLMs) for deployment in enterprise and public sector applications is often guided by performance on English-centric benchmarks. This paper challenges the validity of this approach through a systematic, cross-lingual risk analysis of model reliability. We evaluate eleven prominent open-source LLMs, all runnable on consumer-grade hardware, on a corpus derived from the established MMLU and GSM8K benchmarks. Our fully automated methodology quantifies "performance drift" across UK English (EN), German (DE), and Spanish (ES), employing a novel Hybrid Automated Scoring protocol that uses semantic similarity to objectively assess procedural reasoning against a ground truth.
Our findings reveal a significant "leaderboard illusion": high performance in English is a dangerously poor predictor of a model's capabilities in other languages. We provide quantitative evidence that the top-performing model in our English baseline exhibited the most catastrophic performance degradation in non-English tests, with its normalized score collapsing by over 145 points into negative territory. Furthermore, we demonstrate that this performance drift is not uniform; complex procedural reasoning is significantly more brittle to linguistic shifts than simple factual recall.
Critically, we analyse not just if a model fails, but how it fails. The data reveals distinct "safety fingerprints," where a model's propensity to hallucinate, guess, or honestly admit ignorance changes dramatically with language. This study concludes that the uncritical deployment of LLMs based on monolingual metrics is a high-risk strategy that can lead to the propagation of misinformation and the delivery of inequitable, unreliable services. The accessible methodology presented here offers a necessary, data-driven framework for conducting the essential due diligence required for the safe and responsible use of AI in a multilingual world.


---

## Repository Structure
```bash
llm-multi-lingual-Rubric/
│
├── analysis_outputs # Output files from analyse_results.py
├── data/ # Core data assets
│ └── master_corpus.tsv # The human-editable master list of 350 questions
├── experimental_results # Output from run_experiments.py
├── .gitignore
├── LICENSE
├── papers/ # Final research paper
│ └── cross_lingual_study.pdf
├── README.md
├── requirements.txt # Python dependencies
├── src/ # All Python source code
│ ├── build_corpus.py # Generates the final JSONL corpus from the master TSV
│ ├── translate_corpus.py # Creates and validates multilingual question files
│ ├── run_experiments.py # Executes the tests against the Ollama API
│ ├── score_results.py # Applies the hybrid automated scoring logic
│ └── analyse_results.py # Generates all final tables and visualizations
├── translation_outputs
  └── questions_de_....JSONL, questions_es_....JSONL and translation_log....txt
```


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
    cd llm-multi-lingual-rubric
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
    *This is a potentially time-consuming step.*

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
Hejroe. (2025). The Leaderboard Illusion: A Risk Analysis of Performance Degradation in Multilingual Large Language Models. GitHub Repository. https://github.com/hejroe/Multi-Ling-Rubric
