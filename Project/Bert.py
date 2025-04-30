import pandas as pd
from bert_score import score


df = pd.read_excel('Final_QA.xlsx')
originals = df['original_answer'].astype(str).tolist()
summaries = df['summary_answer'].astype(str).tolist()


# Step 3: Calculate BERT Scores
P, R, F1 = score(summaries, originals, lang="en", verbose=True)

# Step 4: Add the BERTScore outputs to dataframe
df['bert_precision'] = P.tolist()
df['bert_recall'] = R.tolist()
df['bert_f1'] = F1.tolist()

# Step 5: Save the updated dataframe to a new Excel file
df.to_excel('Final_QA_with_BERTScore.xlsx', index=False)

print("Saved updated file with BERT scores!")