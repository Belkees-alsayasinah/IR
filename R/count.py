from collections import defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# قائمة للكلمات الوظيفية
stop_words = set(stopwords.words('english'))

# قائمة للعلامات الترقيمية
punctuation = set(string.punctuation)

def count_word_in_documents(file_path, word):
    # قاموس لحساب عدد المستندات التي تظهر فيها كل كلمة
    word_document_count = defaultdict(int)

    # البدء في قراءة الملف وتحليل النصوص
    with open(file_path, 'r', encoding='utf-8') as file:
        for doc in file:
            # تحويل النص إلى حالة نصية منخفضة وفصله إلى كلمات
            words = word_tokenize(doc.lower())
            # إزالة الكلمات الوظيفية والعلامات الترقيمية
            words = [word for word in words if word not in stop_words and word not in punctuation]
            # إضافة الكلمات إلى القاموس
            if word in words:
                word_document_count[word] += 1

    return word_document_count[word]

# مسار الملف الذي تريد تحليله
file_path = input("Enter the file path: ")

# الكلمة التي تريد البحث عن تكرارها
word = input("Enter the word to search for: ")

# البحث عن تكرار الكلمة في المستندات وطباعة النتيجة
word_count = count_word_in_documents(file_path, word)
print(f"The word '{word}' appeared in {word_count} documents in the file.")

