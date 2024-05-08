import json

# Read data from bengali_test.jsonl
bengali_test_data = []
with open("C:\\Users\\User\\Desktop\\Thesis Work\\datasets\\Bangla - Original\\bengali_test.jsonl", 'r', encoding='utf-8') as file:
    for line in file:
        obj = json.loads(line)
        bengali_test_data.append(obj)

# Read data from human.json
human_data = json.load(open("C://Users//User//Desktop//Thesis Work//datasets//Generated Summaries//summaries_human_8102_mT5_small.json", 'r', encoding='utf-8'))

# Read data from pivot.json
pivot_data = json.load(open("C://Users//User//Desktop//Thesis Work//datasets//Generated Summaries//summaries_pivot_8102_mT5_small.json", 'r', encoding='utf-8'))

# Array containing 30 numbers
numbers = [7, 9, 76, 187, 289, 290, 328, 330, 333, 359, 387, 398, 400, 403, 438, 478, 502, 530, 544, 567, 576, 599, 608, 671, 771, 818, 865, 873, 894, 999]

# Iterate through each number
for num in numbers:
    print(num)
    print(f"TEXT: {bengali_test_data[num - 1]['text']}")
    print(f"HUMAN SUMMARY: {human_data[num - 1]['summary']}")
    print(f"PIVOT SUMMARY: {pivot_data[num - 1]['summary']}")
    print()
