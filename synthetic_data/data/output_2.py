import re

# Function to extract terms from the txt file
def extract_terms(file_path):
    terms = []
    with open(file_path, 'r') as file:
        content = file.read()
        # Regular expression to match the text within {{term|[[ ]]}}
        matches = re.findall(r'\{\{term\|\[\[(.*?)\]\]\}\}', content)
        terms.extend(matches)
    return terms

# Function to write terms to an output file
def write_terms_to_file(terms, output_file_path):
    with open(output_file_path, 'w') as file:
        for term in terms:
            file.write(term + '\n')

# Specify the path to your input txt file
file_path = 'input_2.txt'

# Specify the path for the output txt file
output_file_path = 'output_2_preview.txt'

# Extract terms
terms = extract_terms(file_path)

# Write the extracted terms to output.txt
write_terms_to_file(terms, output_file_path)

print(f"Extracted terms have been saved to {output_file_path}.")
