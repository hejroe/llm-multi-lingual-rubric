# analyse_results.py (Version 2 - Corrected)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

# --- Configuration ---
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")
OUTPUT_DIR = "analysis_outputs"

# --- Dynamic File Finding ---
def find_latest_file(pattern):
    """Finds the most recently created file matching a pattern."""
    try:
        list_of_files = glob.glob(pattern)
        if not list_of_files: return None
        latest_file = max(list_of_files, key=os.path.getctime)
        return latest_file
    except Exception as e:
        print(f"Error finding file for pattern {pattern}: {e}")
        return None

def main():
    """
    Main function to load scored data, perform analysis, and generate outputs.
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # --- Load Data ---
    scored_file = find_latest_file('final_scored_results_*.csv')
    if not scored_file:
        print("Error: No 'final_scored_results_*.csv' file found.")
        return
        
    print(f"--- Starting Analysis on '{os.path.basename(scored_file)}' ---")
    df = pd.read_csv(scored_file)
    
    # --- Data Preparation ---
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    df.dropna(subset=['score'], inplace=True)
    
    models = sorted(df['model_identifier'].unique())
    languages = ['EN', 'DE', 'ES']

    # --- Output 1 & 2: Overall Performance Calculations & CSV ---
    overall_pivot = df.groupby(['model_identifier', 'language'])['score'].sum().unstack()
    max_possible_scores = df.groupby('language')['question_id'].nunique()
    
    for lang in languages:
        if lang in overall_pivot.columns:
            overall_pivot[lang] = (overall_pivot[lang] / max_possible_scores.get(lang, 1)) * 100

    # Ensure all language columns exist before calculating drift
    for lang in languages:
        if lang not in overall_pivot.columns:
            overall_pivot[lang] = 0.0

    overall_pivot['DE_Drift_Pts'] = (overall_pivot['EN'] - overall_pivot['DE'])
    overall_pivot['ES_Drift_Pts'] = (overall_pivot['EN'] - overall_pivot['ES'])
    
    overall_summary_df = overall_pivot.reset_index()
    output_path = os.path.join(OUTPUT_DIR, 'summary_overall_performance.csv')
    overall_summary_df.to_csv(output_path, index=False, float_format='%.2f')
    print(f"\nSaved overall performance summary to '{output_path}'")

    # --- Console Output ---
    print("\n--- High-Level Summary Statistics ---")
    print(f"Total results analyzed: {len(df)}")
    en_highest_model = overall_summary_df.loc[overall_summary_df['EN'].idxmax()]
    print(f"Highest Score (EN): {en_highest_model['model_identifier']} ({en_highest_model['EN']:.2f}%)")
    de_greatest_drift_model = overall_summary_df.loc[overall_summary_df['DE_Drift_Pts'].idxmax()]
    print(f"Greatest Drift (DE): {de_greatest_drift_model['model_identifier']} ({de_greatest_drift_model['DE_Drift_Pts']:.2f} pts)")
    es_greatest_drift_model = overall_summary_df.loc[overall_summary_df['ES_Drift_Pts'].idxmax()]
    print(f"Greatest Drift (ES): {es_greatest_drift_model['model_identifier']} ({es_greatest_drift_model['ES_Drift_Pts']:.2f} pts)")


    # --- Output 3: Domain-Specific Performance CSV ---
    domain_pivot = df.groupby(['model_identifier', 'domain', 'language'])['score'].sum().unstack(fill_value=0)
    max_domain_scores = df.groupby(['domain', 'language'])['question_id'].nunique().unstack(fill_value=0)

    for lang in languages:
        if lang in domain_pivot.columns:
            domain_pivot[lang] = domain_pivot.apply(
                lambda row: (row[lang] / max_domain_scores.loc[row.name[1], lang]) * 100 if max_domain_scores.loc[row.name[1], lang] > 0 else 0,
                axis=1
            )
            
    for lang in languages:
        if lang not in domain_pivot.columns:
            domain_pivot[lang] = 0.0

    domain_pivot['DE_Drift_Pts'] = (domain_pivot['EN'] - domain_pivot['DE'])
    domain_pivot['ES_Drift_Pts'] = (domain_pivot['EN'] - domain_pivot['ES'])
    
    domain_summary_df = domain_pivot.reset_index()
    output_path = os.path.join(OUTPUT_DIR, 'summary_domain_performance.csv')
    domain_summary_df.to_csv(output_path, index=False, float_format='%.2f')
    print(f"Saved domain-specific performance summary to '{output_path}'")
    

    # --- Output 4: Response Category Analysis CSV ---
    category_counts = df.groupby(['model_identifier', 'language', 'score_category']).size().unstack(fill_value=0)
    total_counts = df.groupby(['model_identifier', 'language']).size()
    category_percentages = (category_counts.T / total_counts).T * 100
    
    category_summary_df = category_percentages.reset_index()
    output_path = os.path.join(OUTPUT_DIR, 'summary_category_analysis.csv')
    category_summary_df.to_csv(output_path, index=False, float_format='%.2f')
    print(f"Saved response category analysis to '{output_path}'")


    # --- Visualization Generation ---
    print("\nGenerating visualizations...")
    
    # --- Output 5: Overall Performance Visualization ---
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
    output_path = os.path.join(OUTPUT_DIR, 'figure_1_overall_performance.png')
    plt.savefig(output_path, dpi=300)
    print(f"Saved overall performance chart to '{output_path}'")
    plt.close()


    # --- Output 6: Domain-Specific Drift Visualization (Slope Chart) ---
    plt.figure(figsize=(12, len(models) * 1.5)) # Dynamic height
    
    # Correctly prepare data for slope chart
    slope_data = domain_summary_df.set_index(['model_identifier', 'domain'])
    
    y_pos = 0
    y_ticks = []
    y_labels = []

    for model in models:
        y_pos += 2
        try:
            fa_en = slope_data.loc[(model, 'Factual Accuracy'), 'EN']
            fa_de = slope_data.loc[(model, 'Factual Accuracy'), 'DE']
            pr_en = slope_data.loc[(model, 'Procedural Reasoning'), 'EN']
            pr_de = slope_data.loc[(model, 'Procedural Reasoning'), 'DE']
            
            # Plot FA line
            plt.plot([0, 1], [fa_en, fa_de], marker='o', color='royalblue', linewidth=2)
            # Plot PR line
            plt.plot([0, 1], [pr_en, pr_de], marker='o', color='firebrick', linestyle='--')
            
            plt.text(-0.05, (fa_en + pr_en) / 2, model, ha='right', va='center', weight='bold')
            y_ticks.extend([fa_en, pr_en])
            y_labels.extend([f"{fa_en:.0f}", f"{pr_en:.0f}"])

        except KeyError:
            print(f"Warning: Missing data for model {model} in slope chart.")

    plt.xticks([0, 1], ['English (EN)', 'German (DE)'], fontsize=12)
    plt.xlim(-0.5, 1.5)
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='royalblue', lw=2, label='Factual Accuracy'),
                       Line2D([0], [0], color='firebrick', linestyle='--', lw=2, label='Procedural Reasoning')]
    plt.legend(handles=legend_elements, fontsize=12, loc='best')
    plt.title('Performance Drift: Factual vs. Reasoning (EN to DE)', fontsize=18, weight='bold')
    plt.ylabel('Normalized Score (%)', fontsize=12)
    plt.grid(axis='y')
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'figure_2_domain_drift.png')
    plt.savefig(output_path, dpi=300)
    print(f"Saved domain drift chart to '{output_path}'")
    plt.close()


    # --- Output 7: Response Category Visualization ---
    category_order = ['Correct', 'CorrectProcess_IncorrectResult', 'IDK', 'AmbiguousReasoning', 'IncorrectGuess', 'Fabrication', 'Incorrect']
    category_summary_df_melted = category_summary_df.melt(id_vars=['model_identifier', 'language'], var_name='score_category', value_name='percentage')
    category_summary_df_melted['score_category'] = pd.Categorical(category_summary_df_melted['score_category'], categories=category_order, ordered=True)
    
    g = sns.catplot(
        data=category_summary_df_melted,
        kind='bar',
        x='model_identifier',
        y='percentage',
        hue='score_category',
        col='language',
        col_order=languages,
        height=8,
        aspect=1.5,
        palette='colorblind',
        dodge=False # This creates stacked bars
    )
    g.fig.suptitle('Response Category Distribution by Model and Language', y=1.03, fontsize=18, weight='bold')
    g.set_axis_labels("Model", "Percentage of Responses (%)")
    g.set_xticklabels(rotation=45, ha='right')
    g.tight_layout(rect=[0, 0, 1, 0.97])
    
    output_path = os.path.join(OUTPUT_DIR, f'figure_3_response_categories.png')
    plt.savefig(output_path, dpi=300)
    print(f"Saved combined response category chart to '{output_path}'")
    plt.close()

    print("\n--- Analysis complete. All outputs saved to the 'analysis_outputs' directory. ---")


if __name__ == '__main__':
    main()
