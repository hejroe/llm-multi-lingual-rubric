# analyse_results.py (Version 5 - Final Corrected)

"""
analyse_results.py

Purpose:
  This is the final script in the experimental pipeline. It loads the clean,
  fully scored dataset and generates all the summary tables (CSVs) and
  visualizations (PNGs) required for the research paper.

Inputs:
  - ../analysis_outputs/final_scored_results_[latest].csv: The clean,
    fully scored dataset.

Outputs:
  This script saves the following files into the 'analysis_outputs/' directory:
  - summary_overall_performance.csv
  - summary_domain_performance.csv
  - summary_category_analysis.csv
  - figure_1_overall_performance.png
  - figure_2_domain_drift.png
  - figure_3_response_categories.png
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from pathlib import Path

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = BASE_DIR / 'analysis_outputs'

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")

def find_latest_file(directory, pattern):
    """Finds the most recently created file in a directory matching a pattern."""
    try:
        search_path = os.path.join(directory, pattern)
        list_of_files = glob.glob(str(search_path))
        if not list_of_files: return None
        return max(list_of_files, key=os.path.getctime)
    except Exception as e:
        print(f"Error finding file for pattern {pattern}: {e}")
        return None

def main():
    """
    Main function to load scored data, perform analysis, and generate all outputs.
    """
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
        
    # --- 1. Load Data ---
    scored_file = find_latest_file(ANALYSIS_DIR, 'final_scored_results_*.csv')
    if not scored_file:
        print(f"Error: No 'final_scored_results_*.csv' file found in '{ANALYSIS_DIR}'.")
        return
        
    print(f"--- Starting Analysis on '{os.path.basename(scored_file)}' ---")
    df = pd.read_csv(scored_file)
    
    # --- 2. Data Preparation ---
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    df.dropna(subset=['score'], inplace=True)
    
    models = sorted(df['model_identifier'].unique())
    languages = ['EN', 'DE', 'ES']

    # --- 3. Calculations ---
    # (This entire section is unchanged and correct)
    overall_pivot = df.groupby(['model_identifier', 'language'])['score'].sum().unstack(fill_value=0)
    max_possible_scores = df.groupby('language')['question_id'].nunique()
    for lang in languages:
        if lang in overall_pivot.columns and max_possible_scores.get(lang, 0) > 0:
            overall_pivot[lang] = (overall_pivot[lang] / max_possible_scores[lang]) * 100
    for lang in languages:
        if lang not in overall_pivot.columns:
            overall_pivot[lang] = 0.0
    overall_pivot['DE_Drift_Pts'] = overall_pivot['EN'] - overall_pivot['DE']
    overall_pivot['ES_Drift_Pts'] = overall_pivot['EN'] - overall_pivot['ES']
    overall_summary_df = overall_pivot.reset_index()
    
    domain_pivot = df.groupby(['model_identifier', 'domain', 'language'])['score'].sum().unstack(fill_value=0)
    max_domain_scores = df.groupby(['domain', 'language'])['question_id'].nunique().unstack(fill_value=0)
    for lang in languages:
        if lang in domain_pivot.columns:
            domain_pivot[lang] = domain_pivot.apply(
                lambda row: (row[lang] / max_domain_scores.loc[row.name[1], lang]) * 100 if row.name[1] in max_domain_scores.index and lang in max_domain_scores.columns and max_domain_scores.loc[row.name[1], lang] > 0 else 0,
                axis=1
            )
    domain_pivot['DE_Drift_Pts'] = domain_pivot['EN'] - domain_pivot['DE']
    domain_pivot['ES_Drift_Pts'] = domain_pivot['EN'] - domain_pivot['ES']
    domain_summary_df = domain_pivot.reset_index()
    
    category_counts = df.groupby(['model_identifier', 'language', 'score_category']).size().unstack(fill_value=0)
    total_counts = df.groupby(['model_identifier', 'language']).size()
    category_percentages = (category_counts.T / total_counts).T * 100
    category_summary_df = category_percentages.reset_index()

    # --- 4. Generate Outputs ---
    # (This section is unchanged and correct)
    print("\n--- High-Level Summary Statistics ---")
    print(f"Total results analyzed: {len(df)}")
    if not overall_summary_df.empty and 'EN' in overall_summary_df.columns:
        en_highest_model = overall_summary_df.loc[overall_summary_df['EN'].idxmax()]
        print(f"Highest Score (EN): {en_highest_model['model_identifier']} ({en_highest_model['EN']:.2f}%)")
        de_greatest_drift_model = overall_summary_df.loc[overall_summary_df['DE_Drift_Pts'].idxmax()]
        print(f"Greatest Drift (DE): {de_greatest_drift_model['model_identifier']} ({de_greatest_drift_model['DE_Drift_Pts']:.2f} pts)")
        es_greatest_drift_model = overall_summary_df.loc[overall_summary_df['ES_Drift_Pts'].idxmax()]
        print(f"Greatest Drift (ES): {es_greatest_drift_model['model_identifier']} ({es_greatest_drift_model['ES_Drift_Pts']:.2f} pts)")
    
    overall_summary_df.to_csv(ANALYSIS_DIR / 'summary_overall_performance.csv', index=False, float_format='%.2f')
    print(f"\nSaved overall performance summary to '{ANALYSIS_DIR / 'summary_overall_performance.csv'}'")
    domain_summary_df.to_csv(ANALYSIS_DIR / 'summary_domain_performance.csv', index=False, float_format='%.2f')
    print(f"Saved domain-specific performance summary to '{ANALYSIS_DIR / 'summary_domain_performance.csv'}'")
    category_summary_df.to_csv(ANALYSIS_DIR / 'summary_category_analysis.csv', index=False, float_format='%.2f')
    print(f"Saved response category analysis to '{ANALYSIS_DIR / 'summary_category_analysis.csv'}'")

    print("\nGenerating visualizations...")
    
    # Output 5: Overall Performance Visualization
    plt.figure(figsize=(16, 9))
    plot_df = overall_summary_df.melt(id_vars='model_identifier', value_vars=['EN', 'DE', 'ES'], var_name='Language', value_name='Score')
    sns.barplot(data=plot_df, x='model_identifier', y='Score', hue='Language', errorbar=None)
    plt.title('Overall Performance Score by Model and Language', fontsize=18, weight='bold')
    plt.ylabel('Normalized Score (%)', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Language', fontsize=12)
    plt.tight_layout()
    plt.ylim(bottom=min(0, plot_df['Score'].min() - 10))
    output_path = ANALYSIS_DIR / 'figure_1_overall_performance.png'
    plt.savefig(output_path, dpi=300)
    print(f"Saved overall performance chart to '{output_path}'")
    plt.close()

    # Output 6: Domain-Specific Drift Visualization (Slope Chart)
    plt.figure(figsize=(12, 10))
    slope_df = domain_summary_df.melt(id_vars=['model_identifier', 'domain'], value_vars=['EN', 'DE'], var_name='language', value_name='Score')
    sns.lineplot(data=slope_df, x='language', y='Score', hue='domain', style='domain', markers=True, dashes=False, legend='full', units='model_identifier', estimator=None, lw=0.5, alpha=0.5)
    for model in models:
        en_scores = slope_df[(slope_df['model_identifier'] == model) & (slope_df['language'] == 'EN')]
        if not en_scores.empty:
            y_pos = en_scores['Score'].mean()
            plt.text(-0.05, y_pos, model, ha='right', va='center', fontsize=9, weight='bold')
    plt.xticks(['EN', 'DE'], ['English (EN)', 'German (DE)'], fontsize=12)
    plt.title('Performance Drift by Domain (EN to DE)', fontsize=18, weight='bold')
    plt.ylabel('Normalized Score (%)', fontsize=12)
    plt.xlabel('Language', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=12, loc='best')
    plt.tight_layout()
    output_path = ANALYSIS_DIR / 'figure_2_domain_drift.png'
    plt.savefig(output_path, dpi=300)
    print(f"Saved domain drift chart to '{output_path}'")
    plt.close()

    # Output 7: Response Category Visualization
    category_order = ['Correct', 'CorrectProcess_IncorrectResult', 'IDK', 'AmbiguousReasoning', 'IncorrectGuess', 'Fabrication', 'Incorrect']
    melted_df = category_summary_df.melt(id_vars=['model_identifier', 'language'], 
                                         value_vars=[cat for cat in category_order if cat in category_summary_df.columns],
                                         var_name='score_category', value_name='percentage')
    melted_df.dropna(subset=['percentage'], inplace=True)
    melted_df['score_category'] = pd.Categorical(melted_df['score_category'], categories=category_order, ordered=True)
    
    g = sns.catplot(
        data=melted_df,
        kind='bar',
        x='percentage',
        y='model_identifier',
        hue='score_category',
        col='language',
        col_order=languages,
        height=8,
        aspect=0.8,
        palette='colorblind',
        orient='h',
        dodge=False
    )
    g.fig.suptitle('Response Category Distribution by Model and Language', y=1.03, fontsize=18, weight='bold')
    g.set_axis_labels("Percentage of Responses (%)", "Model")
    g.set_titles("Language: {col_name}")
    # *** THIS IS THE CORRECTED LINE ***
    g.set_yticklabels(rotation=0)
    g.tight_layout(rect=[0, 0, 1, 0.97])

    output_path = ANALYSIS_DIR / 'figure_3_response_categories.png'
    plt.savefig(output_path, dpi=300)
    print(f"Saved combined response category chart to '{output_path}'")
    plt.close()

    print("\n--- Analysis complete. All outputs saved to the 'analysis_outputs' directory. ---")

if __name__ == '__main__':
    main()
