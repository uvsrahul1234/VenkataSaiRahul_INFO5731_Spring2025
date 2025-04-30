import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

df = pd.read_excel('Final_QA.xlsx')  
# This file should have human_score_scaled and automatic metrics like bleu, rouge, bert_f1, etc.

df['human_score_scaled'] = df['Human_score'] / 10

df.to_excel("Final_QA_Scaled.xlsx", index=False)

df = pd.read_excel('Final_QA_Scaled.xlsx')

def compute_pearson(x, y):
    correlation, p_value = pearsonr(x, y)
    return correlation

results = {}

metrics = [
    'bert_f1', 'exact_match_score', 'partial_match_score', 
    'rouge1_f1', 'rouge2_f1', 'rougeL_f1', 'Average_ROUGE_F1', 
    'bleu1', 'bleu2', 'bleu3', 'bleu4', 'Avg_bleu', 
    'llm_based_score'
]

for metric in metrics:
    corr = compute_pearson(df[metric], df['human_score_scaled'])
    results[metric] = corr


correlation_df = pd.DataFrame(list(results.items()), columns=['Metric', 'Pearson Correlation with Human'])
correlation_df = correlation_df.sort_values(by='Pearson Correlation with Human', ascending=False).reset_index(drop=True)

print(correlation_df)

# Identify top metric
top_metric = correlation_df.iloc[0]['Metric']  # assumes sorted descending

# Get blue palette
blue_palette = sns.color_palette("Blues_r", len(correlation_df))

# Highlight top metric in orange, others in blue shades
colors = ['orange' if m == top_metric else blue_palette[i] for i, m in enumerate(correlation_df['Metric'])]

# Plot
plt.figure(figsize=(8, 5))
sns.set_theme(style="whitegrid")

barplot = sns.barplot(
    x="Pearson Correlation with Human", 
    y="Metric", 
    data=correlation_df, 
    palette=colors
)

# Add value labels
for i, (value, name) in enumerate(zip(correlation_df['Pearson Correlation with Human'], correlation_df['Metric'])):
    plt.text(value + 0.01, i, f"{value:.2f}", va='center', fontweight='bold', fontsize=8)

# Labels and styling
plt.title("Pearson Correlation of Metrics with Human Judgments", fontsize=16, fontweight='bold')
plt.xlabel("Pearson Correlation Coefficient", fontsize=14)
plt.ylabel("Evaluation Metric", fontsize=14)
plt.xlim(0, 1)
plt.tight_layout()
plt.grid(False)
plt.show()


## Overlapping Histogram

# Load your data
df = pd.read_excel("Final_QA_Scaled.xlsx")

# Drop NaNs in the relevant columns
df = df[['human_score_scaled', 'llm_based_score']].dropna()

# Define bins (0.0–1.0 in 0.1 steps)
bins = [i/10 for i in range(11)]  # [0.0, 0.1, ..., 1.0]

# Plot overlapping histograms
plt.figure(figsize=(8, 5))

plt.hist(
    df['human_score_scaled'],
    bins=bins,
    alpha=0.5,
    label='Human Evaluation',
    color='blue',
    edgecolor='black'
)

plt.hist(
    df['llm_based_score'],
    bins=bins,
    alpha=0.5,
    label='LLM-Based Evaluation',
    color='orange',
    edgecolor='black'
)

# Labels and legend
plt.title("Overlapping Histogram of Human vs LLM-Based Evaluation Scores", fontsize=14, fontweight='bold')
plt.xlabel("Score Bin")
plt.ylabel("Number of Answers")
plt.legend()
plt.xticks(bins)
plt.tight_layout()
plt.yticks([])
plt.show()
