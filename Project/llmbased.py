import pandas as pd
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Setup your Groq API Key
os.environ["GROQ_API_KEY"] = "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"  # Replace with your actual key

# Define the scorer class
class GroqAnswerSimilarityScorer:
    def __init__(self):
        self.model = ChatGroq(
            model_name="llama-3.1-8b-instant",
            temperature=0.1,
            max_tokens=100
        )
        self.parser = JsonOutputParser()

        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a similarity scoring assistant. Given two answers, score their semantic similarity on a scale from 0 (completely different) to 1 (identical meaning). 
        Return only JSON in this format: {{"similarity_score": <float between 0 and 1>}}"""),
            ("human", "Answer A: {answer_a}\nAnswer B: {answer_b}")
        ])

        self.chain = self.prompt_template | self.model | self.parser

    def get_similarity(self, answer_a: str, answer_b: str) -> float:
        """
        Returns similarity score between 0 and 1 using Groq API
        """
        try:
            response = self.chain.invoke({
                "answer_a": answer_a,
                "answer_b": answer_b
            })
            return float(response["similarity_score"])
        except Exception as e:
            print(f"⚠️ Error scoring pair: {e}")
            return 0.0  # Fallback if any error occurs

# === Main Section ===
if __name__ == "__main__":
    # Load your dataset
    df = pd.read_excel("Final_QA.xlsx")  # Replace with your file
    df = df.head(2000)
    # Initialize scorer
    scorer = GroqAnswerSimilarityScorer()

    similarity_scores = []

    # Loop through your rows
    for idx, row in df.iterrows():
        sentence_a = row['original_answer']    # First column
        sentence_b = row['summary_answer']   # Second column

        # Get similarity score
        score = scorer.get_similarity(sentence_a, sentence_b)
        similarity_scores.append(score)

        if (idx + 1) % 10 == 0:
            print(f"✅ Processed {idx+1} pairs")

    # Save scores back to DataFrame
    df['similarity_score'] = similarity_scores

    # Save the new dataset
    df.to_csv("similarity_scored_output.csv", index=False)

    print("\n🎯 Finished! Saved to similarity_scored_output.csv ✅")

