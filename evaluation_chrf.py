import json
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import evaluate

# Load reference summaries from bengali_test.jsonl
reference_summaries = []
with open("D:\\Thesis Work\\datasets\\Bangla - Original\\bengali_test.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        reference_summaries.append(data["summary"])

# Load summaries from summaries_after_2000.json
with open("D:\\Thesis Work\\datasets\\Generated Summaries\\summaries_pivot_8102_mT5_small.json", "r", encoding="utf-8") as f:
    summaries = json.load(f)

# Initialize CHRF++ scorer
chrf = evaluate.load('chrf')

# Calculate CHRF++ scores and add them to the data
total_chrf_score = 0
for summary in tqdm(summaries, desc="Calculating CHRF++ Scores"):
    generated_summary = summary["summary"]
    chrf_score = chrf.compute(predictions=[generated_summary], references=[reference_summaries])['score']
    summary["chrf++_score"] = chrf_score
    total_chrf_score += chrf_score

# Calculate average CHRF++ score
average_chrf_score = total_chrf_score / len(summaries)

# Save the updated data back to summaries_after_2000.json
with open("D:\\Thesis Work\\datasets\\Generated Summaries\\summaries_pivot_8102_mT5_small.json", "w", encoding="utf-8") as f:
    json.dump(summaries, f, ensure_ascii=False, indent=4)

print("Average CHRF++ Score:", average_chrf_score)
