import pandas as pd
import os
from sklearn.model_selection import train_test_split

def convert_arabic_paraphrase_dataset():
    """
    Convert Arabic paraphrasing benchmark-Marwah-Alian.csv to the required format.
    Split into dev.csv (90%) and test.csv (10%).

    Input format: First sentence;second sentence;44_experts;similarity;parahrase
    Output format: sentence1,sentence2,label (where label: p->1, np->0)
    """

    # Input and output file paths
    input_file = "D:\\repos\language_discovery_pruning\\benchmark_data\Arabic-Paraphrasing-Benchmark\Arabic paraphrasing benchmark-Marwah-Alian.csv"
    output_dir = "D:\\repos\language_discovery_pruning\\benchmark_data\Arabic-Paraphrasing-Benchmark"
    dev_file = os.path.join(output_dir, "dev.csv")
    test_file = os.path.join(output_dir, "test.csv")

    # Read the input CSV with semicolon delimiter
    df = pd.read_csv(input_file, delimiter=';')

    # Create new dataframe with required columns
    converted_df = pd.DataFrame()
    converted_df['sentence1'] = df['First sentence']
    converted_df['sentence2'] = df['second sentence']

    # Convert paraphrase labels: 'p' -> 1, 'np' -> 0
    converted_df['label'] = df['parahrase'].map({'p': 1, 'np': 0})

    # Split data: 90% dev, 10% test
    dev_df, test_df = train_test_split(
        converted_df,
        test_size=0.1,
        random_state=42,
        stratify=converted_df['label']  # Ensure balanced split
    )

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the split datasets
    dev_df.to_csv(dev_file, index=False)
    test_df.to_csv(test_file, index=False)

    print(f"Conversion completed successfully!")
    print(f"Original dataset shape: {df.shape}")
    print(f"Converted dataset shape: {converted_df.shape}")
    print(f"\n{'='*50}")
    print(f"Dev set shape: {dev_df.shape}")
    print(f"Test set shape: {test_df.shape}")
    print(f"Dev set saved to: {dev_file}")
    print(f"Test set saved to: {test_file}")

    # Display first few rows of dev set
    print(f"\n{'='*50}")
    print("\nFirst 5 rows of dev dataset:")
    print(dev_df.head())

    # Display label distribution for overall dataset
    print(f"\n{'='*50}")
    print(f"Overall Label distribution:")
    print(converted_df['label'].value_counts().sort_index())
    print(f"Percentages:")
    print(converted_df['label'].value_counts(normalize=True).sort_index() * 100)

    # Display label distribution for dev set
    print(f"\n{'='*50}")
    print(f"Dev set Label distribution:")
    print(dev_df['label'].value_counts().sort_index())
    print(f"Percentages:")
    print(dev_df['label'].value_counts(normalize=True).sort_index() * 100)

    # Display label distribution for test set
    print(f"\n{'='*50}")
    print(f"Test set Label distribution:")
    print(test_df['label'].value_counts().sort_index())
    print(f"Percentages:")
    print(test_df['label'].value_counts(normalize=True).sort_index() * 100)

    return dev_df, test_df

if __name__ == "__main__":
    convert_arabic_paraphrase_dataset()
