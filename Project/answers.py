import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Fix CUDA memory
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Load your local LLaMA model once
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load your cleaned Excel file
df = pd.read_excel("cleaned_questions.xlsx")  # update if needed
df = df.head(2)

# Settings
max_input_tokens = 1024
max_answer_tokens = 50

def clean_generated_answer(answer_text):
    """
    Cleans up extra instruction leakage from generated answer.
    """

    # Stop if you detect prompt-like phrases leaking back
    bad_phrases = [
        "Answer is concise",
        "based on the most relevant information",
        "ignore unrelated details",
        "answer is short",
        "answer is paraphrased",
        "Answer:"
    ]

    for phrase in bad_phrases:
        if phrase.lower() in answer_text.lower():
            # Cut off anything after the bad phrase
            answer_text = answer_text.split(phrase)[0].strip()

    # Optionally stop after 1 or 2 sentences (stronger cleanup)
    sentences = answer_text.split('.')
    if len(sentences) > 2:
        answer_text = '.'.join(sentences[:2]) + '.'

    return answer_text.strip()


def format_messages(messages):
    """
    Converts structured messages into a single text prompt.
    """
    formatted = ""
    role_mapping = {"system": "System", "user": "User", "assistant": "Assistant"}

    for message in messages:
        role = role_mapping.get(message["role"], "User")
        content = message["content"]
        formatted += f"{role}: {content}\n"
    formatted += "Assistant:"  # Assistant is expected to continue
    return formatted

def generate_answer_from_messages(messages):
    """
    Generates an answer from structured messages.
    """
    prompt = format_messages(messages)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens)

    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    input_length = input_ids.shape[1]

    generated = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_answer_tokens,
        temperature=0.5,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    # Only decode newly generated tokens
    generated_ids = generated[0][input_length:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    answer = clean_generated_answer(answer)

#    words = answer.split()
#    if len(words) > 40:
#        answer = " ".join(words[:40]) + "..."

    return answer

# 📝 Now pull from your dataset
df = pd.read_excel("cleaned_questions.xlsx")  # or your current dataset
df = df.iloc[965:]

# 📝 Prepare answer columns
original_answers = []
summary_answers = []

# 📝 Loop over each row
for idx, row in df.iterrows():
    paragraph = row['original_text']
    ref_summary = row['ref_sum']
    question = row['question']

    # Truncate paragraph safely
    tokenized = tokenizer(paragraph, truncation=True, max_length=max_input_tokens // 2, return_tensors="pt")
    truncated_paragraph = tokenizer.decode(tokenized['input_ids'][0], skip_special_tokens=True)

    # Build messages for paragraph
    messages_paragraph = [
        {"role": "system", "content": "You are a helpful assistant who answers questions briefly and accurately based only on the provided context. Make sure the answer is concise and paraphrase if necessary. Answer the question based only on the most relevant information from the provided context. Ignore unrelated details. Keep the answer short"},
        {"role": "user", "content": f"Context: {truncated_paragraph}\n\nQuestion: {question}"}
    ]

    # Build messages for ref_summary
    messages_summary = [
        {"role": "system", "content": "You are a helpful assistant who answers questions briefly and accurately based only on the provided summary. Make sure the answer is concise and paraphrase if necessary"},
        {"role": "user", "content": f"Context: {ref_summary}\n\nQuestion: {question}"}
    ]

    # Generate answers
    original_answer = generate_answer_from_messages(messages_paragraph)
    summary_answer = generate_answer_from_messages(messages_summary)

    original_answers.append(original_answer)
    summary_answers.append(summary_answer)

    # Progress print every 10 rows
    if (idx + 1) % 10 == 0:
        print(f"✅ Processed {idx+1} questions")

# 📝 Save back to DataFrame
df["original_answer"] = original_answers
df["summary_answer"] = summary_answers

# 📝 Save to output
df.to_csv("generated_answers_with_messages.csv", index=False)

print("\n🎯 Processing complete. Saved to generated_answers_with_messages.csv")
