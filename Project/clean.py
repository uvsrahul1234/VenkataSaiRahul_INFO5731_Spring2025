import pandas as pd

# Step 1: Load the questions file
df = pd.read_csv("generated_questions.csv")  # Update with your file path

# Step 2: Remove exact duplicate questions
df_cleaned = df.drop_duplicates(subset=['question'])

df.dropna(inplace=True)

# Step 3: Keep only questions that end with '?'
df = df[df['question'].str.strip().str.endswith('?')]

# Step 4: Save the cleaned file
output_path = "cleaned_questions.csv"
df_cleaned.to_csv(output_path, index=False)

print(f"Cleaned file saved at: {output_path}")


