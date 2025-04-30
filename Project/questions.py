import pandas as pd
import torch, os, re
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc

# Load data
df = pd.read_excel("dm_1000.xlsx")
document_texts = df[['original_doc', 'ref_sum']].iloc[99:].values.tolist()  # use head(10) for testing

# Load tokenizer and model
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Move model to GPU if available
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model.to(device)

# Settings
max_input_tokens = 1024
all_questions = []

for i, (paragraph, ref_sum) in enumerate(document_texts, start=99):
    # Truncate paragraph
    tokenized = tokenizer(paragraph, truncation=True, max_length=max_input_tokens // 2, return_tensors="pt")
    truncated_paragraph = tokenizer.decode(tokenized['input_ids'][0], skip_special_tokens=True)

    # Construct prompt
    prompt = f"""
    Given the following document text, generate 15 questions of varying complexity. The questions should be of type who, what, when, where, why and how. Format as:
    Q1: [Question]
    Q2: [Question]
    ...
    Q15: [Question]

    ### Document:
    {truncated_paragraph}
    """

    # Tokenize prompt and move to device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    # inputs = {k: v.to(model.device) for k, v in inputs.items()}
    #first_device = next(iter(model.hf_device_map.values()))
    #if isinstance(first_device, dict):
        # If you're using 4bit/bnb config, hf_device_map might be nested
     #   first_device = list(first_device.values())[0]

    #inputs = {k: v.to(first_device) for k, v in inputs.items()}
#    inputs = {k: v.to(model.hf_device_map['model.embed_tokens']) for k, v in inputs.items()}

    # Generate output
    output = model.generate(
        **inputs,
        max_new_tokens=400,
        temperature=0.5,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id  # prevents warning
    )

    # Decode result
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Parse individual questions
    lines = generated_text.split("\n")
#    questions = [line.strip() for line in lines if line.strip().startswith("Q")]
    
    questions = []

    for line in lines:
        line = line.strip()
        if re.match(r'^Q\d+:', line):  # If line starts with Q[number]:
            question_text = line.split(":", 1)[1].strip()  # Keep only text after ":"
            questions.append(question_text)

    # Add to result list
#    all_questions.append({
 #       "paragraph_id": i + 1,
  #      "original_text": paragraph,
   #     "ref_sum": ref_sum,
    #    "questions": questions
#    })
    
    # Save all questions for the paragraph
    for j, question in enumerate(questions):
        all_questions.append({
            "paragraph_id": i + 1,
            "original_text": paragraph,
            "ref_sum": ref_sum,
            "question": question
        })

    # Clear memory (optional)
    print(f"✅ Finished generating questions for Paragraph {i+1}\n")
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()

# Convert to DataFrame
#questions_df = pd.DataFrame([
#    {
#        "paragraph_id": item["paragraph_id"],
#        "original_text": item["original_text"],
#        "question": q
#    }
#    for item in all_questions
#    for j, q in enumerate(item["questions"])
#])

# Convert results into DataFrame
questions_df = pd.DataFrame(all_questions)

# Save to CSV
questions_df.to_csv("generated_questions.csv", index=False)
print("✅ Questions saved to 'generated_questions.csv'")

