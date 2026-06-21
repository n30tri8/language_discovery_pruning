import pandas as pd
import os

# Path to the MMLU English benchmark CSV
csv_path = os.path.join(os.path.dirname(__file__), '../benchmark_data/mmlu/en.csv')

# Load the CSV file
df = pd.read_csv(csv_path)

# Find the top three subjects with the most questions
subject_counts = df['Subject'].value_counts()
top_three = subject_counts.head(3)

print("Top 3 subjects by question count:")
for subject, count in top_three.items():
    print(f"Subject: {subject}, Count: {count}")

# Calculate the average number of questions per subject
average_questions = subject_counts.mean()
print(f"\nAverage number of questions per subject: {average_questions:.2f}")

# Calculate question count for specific subjects
SUBJECTS = [
    "philosophy", "international_law", "high_school_mathematics", "professional_psychology",
    "professional_medicine", "sociology", "marketing", "high_school_chemistry", "clinical_knowledge"
]
print("\nQuestion count for selected subjects:")
for subject in SUBJECTS:
    count = subject_counts.get(subject, 0)
    print(f"Subject: {subject}, Count: {count}")


