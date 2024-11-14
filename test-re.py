import re

def title_case_all_upper_phrases(text):
    # Split text by spaces but retain delimiters like punctuation
    parts = re.split(r'(\W+)', text)  # Split on non-word characters but keep them

    transformed_parts = []
    for part in parts:
        # Only transform if part is all-uppercase and doesn't contain non-Latin characters
        if part.isupper() and all(char.isalpha() or char.isspace() for char in part):
            transformed_parts.append(part.title())
        else:
            transformed_parts.append(part)

    return ''.join(transformed_parts)

# Example usage
text = "暗号資産を使ったエンターテインメントプロジェクト「Tokyo Beast」です。"
transformed_text = title_case_all_upper_phrases(text)
print(transformed_text)
