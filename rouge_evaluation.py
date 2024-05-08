import json
from rouge import Rouge 

def evaluate_summary(hypothesis, reference):
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference)
    f1_scores = {}
    for score in scores:
        for metric, values in score.items():
            f1_scores[metric] = values.get('f', 0.0)  # Use get() to handle missing 'f' key gracefully
    return f1_scores

# Load reference summaries from the JSONL file
reference_summaries = []
with open("C:\\Users\\User\\Desktop\\Thesis Work\\datasets\\Bangla - Original\\bengali_test.jsonl", "r", encoding="utf-8") as jsonl_file:
    for line in jsonl_file:
        reference_data = json.loads(line)
        reference_summaries.append(reference_data["summary"])

# Load hypothesis from the JSON file and evaluate ROUGE scores for each summary
with open("C:\\Users\\User\\Desktop\\Thesis Work\\datasets\\Generated Summaries\\summaries_human_8102.json", "r", encoding="utf-8") as json_file:
    data = json.load(json_file)
    for i, obj in enumerate(data):
        hypothesis = obj["summary"]
        reference = reference_summaries[i] if i < len(reference_summaries) else ""  # Ensure we have a reference for each hypothesis
        scores = evaluate_summary(hypothesis, reference)
        # Add ROUGE scores as new columns to the JSON object
        obj["rouge-1"] = scores["rouge-1"]
        obj["rouge-2"] = scores["rouge-2"]
        obj["rouge-L"] = scores["rouge-l"]

# Save the updated data back to the JSON file
with open("C:\\Users\\User\\Desktop\\Thesis Work\\datasets\\Generated Summaries\\summaries_human_8102.json", "w", encoding="utf-8") as json_file:
    json.dump(data, json_file, ensure_ascii=False, indent=4)
