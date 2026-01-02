import pandas as pd
import os

def convert_arabic_paraphrase_dataset():
    """
    Convert Arabic paraphrasing benchmark-Marwah-Alian.csv to the required format.

    Input format: First sentence;second sentence;44_experts;similarity;parahrase
    Output format: sentence1,sentence2,label (where label: p->1, np->0)
    """

    # Input and output file paths
    input_file = "D:\\repos\language_discovery_pruning\\benchmark_data\Arabic-Paraphrasing-Benchmark\Arabic paraphrasing benchmark-Marwah-Alian.csv"
    output_file = "D:\\repos\language_discovery_pruning\\benchmark_data\Arabic-Paraphrasing-Benchmark\\ar-paraphrase.csv"

    # Read the input CSV with semicolon delimiter
    df = pd.read_csv(input_file, delimiter=';')

    # Create new dataframe with required columns
    converted_df = pd.DataFrame()
    converted_df['sentence1'] = df['First sentence']
    converted_df['sentence2'] = df['second sentence']

    # Convert paraphrase labels: 'p' -> 1, 'np' -> 0
    converted_df['label'] = df['parahrase'].map({'p': 1, 'np': 0})

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save the converted dataset
    converted_df.to_csv(output_file, index=False)

    print(f"Conversion completed successfully!")
    print(f"Original dataset shape: {df.shape}")
    print(f"Converted dataset shape: {converted_df.shape}")
    print(f"Output saved to: {output_file}")

    # Display first few rows
    print("\nFirst 5 rows of converted dataset:")
    print(converted_df.head())

    # Display label distribution
    print(f"\nLabel distribution:")
    print(converted_df['label'].value_counts())

    return converted_df

if __name__ == "__main__":
    convert_arabic_paraphrase_dataset()
