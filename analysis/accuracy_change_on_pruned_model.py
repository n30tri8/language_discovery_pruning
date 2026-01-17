import pandas as pd


def process_model_data(raw_model_eval_file_path, cross_benchmark_file_path, output_file_path):
    # 1. Load the datasets
    df_raw = pd.read_csv(raw_model_eval_file_path)
    df_logs = pd.read_csv(cross_benchmark_file_path)

    # 2. Prepare for merging
    # Map column names in cross_benchmark_logs.csv to match the target schema and join keys
    # subject=benchmark, lang=language
    df_logs_renamed = df_logs.rename(columns={
        'benchmark': 'subject',
        'language': 'lang',
        'accuracy': 'accuracy_on_pruned'
    })

    # Rename original accuracy column for clarity
    df_raw_renamed = df_raw.rename(columns={'subtask_acc': 'original_accuracy'})

    # 3. Merge the DataFrames
    # Use (lang, subject) as the composite key
    merged_df = pd.merge(
        df_raw_renamed[['subject', 'lang', 'original_accuracy']],
        df_logs_renamed[['subject', 'lang', 'accuracy_on_pruned']],
        on=['subject', 'lang'],
        how='inner'
    )

    # 4. Compute relative change
    # Formula: relative drop (%) = (original - pruned) / original * 100
    merged_df['absolute_drop'] = round(merged_df['original_accuracy'] - merged_df['accuracy_on_pruned'], 6)
    merged_df['relative_change'] = round((merged_df['absolute_drop'] / merged_df['original_accuracy']) * 100, 3)

    # 5. Finalize columns
    # Target columns: lang, subject, original_accuracy, accuracy_on_pruned, relative_change
    final_df = merged_df[
        ['lang', 'subject', 'original_accuracy', 'accuracy_on_pruned', 'absolute_drop', 'relative_change']]

    # 6. Sort by lang and subject (New Step)
    final_df = final_df.sort_values(by=['lang', 'subject'])

    # 7. Save to CSV
    final_df.to_csv(output_file_path, index=False)
    print(f"File saved to {output_file_path}")


# Example usage
if __name__ == "__main__":
    raw_model_eval_file_path = 'D:\\repos\\language_discovery_pruning\\results\\17-1_llama60p_qwen75\\raw_model_eval.csv'
    cross_benchmark_file_path = 'D:\\repos\language_discovery_pruning\\results\\17-1_llama60p_qwen75\cross_benchmark_logs.csv'
    output_file_path = 'D:\\repos\\language_discovery_pruning\\results\\17-1_llama60p_qwen75\\Llama-3.1-8B_60p_accuracy_comparison.csv'
    process_model_data(raw_model_eval_file_path, cross_benchmark_file_path, output_file_path)
