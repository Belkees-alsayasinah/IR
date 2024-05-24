import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from scipy.spatial.distance import cosine
import os
import spacy

# تحميل المحرك اللغوي الإنجليزي القياسي من spacy
nlp = spacy.load("en_core_web_sm")

# تحميل قائمة الكلمات الوقفية من nltk
stop_words = set(stopwords.words('english'))

# قراءة المتجهات من ملف pkl
vectors_path = r'C:\Users\sayas\.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\spaCy\vectors.pkl'
with open(vectors_path, "rb") as f:
    saved_vectors = pickle.load(f)

# تحميل البيانات النصية من ملف
data_file = r'C:\Users\sayas\.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\collection.tsv'
with open(data_file, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# استخراج النصوص من البيانات
documents = []
for line in lines:
    parts = line.split('\t')
    if len(parts) > 1:
        documents.append(parts[1].strip())

# 1. تنظيف الاستعلام
query = "how to tell the difference between a girl and boy bearded dragon"
query_tokens = word_tokenize(query.lower())
cleaned_query_tokens = [token for token in query_tokens if token.isalnum() and token not in stop_words]
cleaned_query_text = " ".join(cleaned_query_tokens)

# 2. تحويل الاستعلام إلى متجه نصي
query_vector = nlp(cleaned_query_text).vector

# التحقق من طول المتجه النصي للاستعلام
print(f"Query vector length: {len(query_vector)}")

# التحقق من طول المتجهات المحفوظة
print(f"Saved vectors length: {len(saved_vectors)}")
print(f"Length of first saved vector: {len(saved_vectors[0]) if saved_vectors else 'No vectors loaded'}")

# 3. البحث عن الاستعلام في المتجهات المحفوظة
most_similar_index = None
min_distance = float('inf')
for i, saved_vector in enumerate(saved_vectors):
    # التحقق من أن المتجه ليس فارغاً
    if len(saved_vector) == 0:
        print(f"Empty vector at index {i}")
        continue

    # التحقق من أن أبعاد المتجهات متطابقة
    if len(saved_vector) != len(query_vector):
        print(f"Vector length mismatch at index {i}: {len(saved_vector)}")
        continue

    distance = cosine(query_vector, saved_vector)
    if distance < min_distance:
        min_distance = distance
        most_similar_index = i

# التأكد من العثور على مستند مشابه
if most_similar_index is not None and most_similar_index < len(documents):
    # استرجاع المستند الأكثر تشابها من قائمة المستندات
    most_similar_document = documents[most_similar_index]
    print("Most similar document:")
    print(most_similar_document)
else:
    print("No similar document found.")
