import os
import re

def create_or_update_file(file_path, content):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)
    print(f"Created/Updated: {file_path}")

def extract_classes(content):
    class_pattern = r'(class\s+\w+[\s\S]*?(?=\n\n|$))'
    return re.findall(class_pattern, content)

def extract_functions(content):
    function_pattern = r'(def\s+\w+[\s\S]*?(?=\n\n|$))'
    return re.findall(function_pattern, content)

def update_memoraith(base_path, content):
    # Extract file contents
    file_pattern = r'# (memoraith/.*?\.py)\n\n([\s\S]*?)(?=\n# memoraith/|\Z)'
    matches = re.findall(file_pattern, content)

    for file_path, file_content in matches:
        full_path = os.path.join(base_path, file_path)

        # Extract classes and functions
        classes = extract_classes(file_content)
        functions = extract_functions(file_content)

        # Combine classes and functions
        components = classes + functions

        # Create or update the file
        create_or_update_file(full_path, "\n\n".join(components))

if __name__ == "__main__":
    base_path = r"C:\Users\PC\Desktop\Leo-Major\Memoraith"
    with open(r"C:\Users\PC\Desktop\Leo-Major\alop.txt", 'r', encoding='utf-8') as f:
        content = f.read()

    update_memoraith(base_path, content)
    print("Memoraith project has been updated successfully!")