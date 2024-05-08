import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from tqdm.auto import tqdm
import json

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "csebuetnlp/banglat5_nmt_bn_en"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

input_jsonl_path = 'C:\\Users\\User\\Desktop\\Thesis Work\\datasets\\Bangla - Original\\bengali_train.jsonl'  # Adjust with your actual file path
output_jsonl_path = 'C:\\Users\\User\\Desktop\\Thesis Work\\datasets\\Pivot Files\\bengali_train_text_translated_2.jsonl'  # Updated output path for clarity

start_entry = 3433  # Adjust this to start from entry 501 (0-indexed, so use 500 for the 501st entry)
end_entry = 8102  # This remains as the stopping point (inclusive for the 1000th entry)

# Adjust the file reading logic to process the second set of 500 entries
with open(input_jsonl_path, 'r', encoding='utf-8') as input_file:
    lines = [line for i, line in enumerate(input_file) if start_entry <= i < end_entry]  # Adjusted range

with open(output_jsonl_path, 'w', encoding='utf-8') as output_file:
    for line in tqdm(lines, total=len(lines), desc="Translating", unit="entry"):
        data = json.loads(line)
        bengali_text = data.get("text", "")
        sentences = bengali_text.split('।')
        translated_paragraph = []

        for sentence in sentences:
            if sentence.strip() == "":
                continue
            input_ids = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids.to(device)
            generated_tokens = model.generate(input_ids, max_length=512)
            decoded_tokens = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            translated_paragraph.append(decoded_tokens)
        
        final_translated_paragraph = " ".join(translated_paragraph).strip()
        if not final_translated_paragraph.endswith('.'):
            final_translated_paragraph += '.'
        
        json.dump({"text": final_translated_paragraph}, output_file)
        output_file.write('\n')

print("done")