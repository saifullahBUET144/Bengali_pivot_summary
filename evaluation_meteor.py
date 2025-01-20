import json
from nltk.tokenize import word_tokenize
from nltk.translate import meteor_score
from tqdm import tqdm

# Load reference summaries from bengali_test.jsonl
reference_summaries = []
with open("D:\\Thesis Work\\datasets\\Bangla - Original\\bengali_test.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        reference_summaries.append(word_tokenize(data["summary"]))

# Load summaries from summaries_after_2000.json
with open("D:\\Thesis Work\\datasets\\Generated Summaries\\summaries_pivot_8102_mT5_small.json", "r", encoding="utf-8") as f:
    summaries = json.load(f)

# Calculate Meteor scores and add them to the data
total_meteor_score = 0
for summary in tqdm(summaries, desc="Calculating Meteor Scores"):
    summary_tokens = word_tokenize(summary["summary"])
    meteor = meteor_score.meteor_score(reference_summaries, summary_tokens)
    summary["meteor_score"] = meteor
    total_meteor_score += meteor

# Calculate average Meteor score
average_meteor_score = total_meteor_score / len(summaries)

# Save the updated data back to summaries_after_2000.json
with open("D:\\Thesis Work\\datasets\\Generated Summaries\\summaries_pivot_8102_mT5_small.json", "w", encoding="utf-8") as f:
    json.dump(summaries, f, ensure_ascii=False, indent=4)

print("Average Meteor Score:", average_meteor_score)
