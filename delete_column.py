import json

def delete_column(json_file_path, column_name):
    with open(json_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    
    for entry in data:
        if column_name in entry:
            del entry[column_name]
    
    with open(json_file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

# Example usage:
json_file_path = "path/to/your/json/file.json"
column_name_to_delete = "column_name_to_delete"

delete_column(json_file_path, column_name_to_delete)
print(f"Column '{column_name_to_delete}' has been deleted from the JSON file.")
