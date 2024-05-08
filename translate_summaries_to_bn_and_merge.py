import json
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "csebuetnlp/banglat5_nmt_en_bn"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

input_json_path = 'C:\\Users\\User\\Desktop\\Thesis Work\\datasets\\Pivot Files\\english_pivot_with_summmaries.json'
original_jsonl_path = 'C:\\Users\\User\\Desktop\\Thesis Work\\datasets\\Bangla - Original\\bengali_train.jsonl'
updated_jsonl_path = 'C:\\Users\\User\\Desktop\\Thesis Work\\datasets\\bengali_train_with_pivot_summary.jsonl'

# Load the JSON file with summaries to translate
with open(input_json_path, 'r', encoding='utf-8') as file:
    summaries_data = json.load(file)

# Define the start and end entry for processing
start_entry = 0
end_entry = 8102  # Adjust as needed, ensuring not to exceed list bounds

# Translate summaries
translated_summaries = []
for entry in tqdm(summaries_data[start_entry:end_entry], desc="Translating", unit="entry"):
    english_text = entry.get("summary", "")
    input_ids = tokenizer(english_text, return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids.to(device)
    generated_tokens = model.generate(input_ids, max_length=512)
    decoded_tokens = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    final_translated_summary = decoded_tokens.strip()
    translated_summaries.append(final_translated_summary)

# Update the original JSONL file with translated summaries in the specified range
with open(original_jsonl_path, 'r', encoding='utf-8') as original_file, \
     open(updated_jsonl_path, 'w', encoding='utf-8') as updated_file:
    for i, line in enumerate(original_file):
        if start_entry <= i < end_entry:
            entry = json.loads(line)
            entry["pivot_summary"] = translated_summaries[i-start_entry]  # Use adjusted index
            json.dump(entry, updated_file, ensure_ascii=False)
        else:
            updated_file.write(line)  # Write unmodified line for entries outside the range
        updated_file.write('\n')

print("Updated JSONL file with 'pivot_summary' column has been created.")
