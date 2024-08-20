import json

def count_json_items(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return len(data)

file1 = "baseline_proba2go.txt"
file2 = "new_baseline.txt"

count1 = count_json_items(file1)
count2 = count_json_items(file2)

print(f"Number of items in {file1}: {count1}")
print(f"Number of items in {file2}: {count2}")
