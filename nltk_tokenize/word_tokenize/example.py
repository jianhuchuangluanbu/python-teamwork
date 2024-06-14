import nltk
from nltk.tokenize import word_tokenize
from test_cases import word_tokenize_cases

# nltk.download('punkt')

for case in word_tokenize_cases:
    text = case["text"]
    result = word_tokenize(text)
    print(result)