import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pickle
#حاولة فاشلة لتطبيق Spacy

nlp = spacy.load("en_core_web_md")

data_file = r'C:\Users\sayas\.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\new_processed_collection.tsv'
with open(data_file, 'r', encoding='utf-8') as file:
    documents = file.readlines()


stop_words = set(stopwords.words('english'))


tokenized_documents = []
for doc in documents:
    tokens = word_tokenize(doc.lower())
    cleaned_tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    if cleaned_tokens:
        tokenized_documents.append(cleaned_tokens)


vectors = []
for doc_tokens in tokenized_documents:
    doc_text = " ".join(doc_tokens)
    if doc_text:
        doc_vector = nlp(doc_text).vector
        vectors.append(doc_vector)
    else:
        vectors.append([])

# حفظ المتجهات النصية في ملف pkl
with open("vectors.pkl", "wb") as f:
    pickle.dump(vectors, f)
