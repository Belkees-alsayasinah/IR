import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pickle

# تحميل المحرك اللغوي الإنجليزي القياسي من spacy
nlp = spacy.load("en_core_web_md")  # استخدام نسخة أكبر من النموذج إذا كانت متاحة للحصول على دقة أفضل

# تحميل البيانات النصية من ملف
data_file = r'C:\Users\sayas\.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\new_processed_collection.tsv'
with open(data_file, 'r', encoding='utf-8') as file:
    documents = file.readlines()

# تحميل قائمة الكلمات الوقفية من nltk
stop_words = set(stopwords.words('english'))

# تنظيف وتحويل البيانات النصية إلى قوائم كلمات باستخدام NLTK و spaCy
tokenized_documents = []
for doc in documents:
    tokens = word_tokenize(doc.lower())  # تحويل النص إلى حالة صغيرة وتقسيمه إلى كلمات
    # إزالة الكلمات الوقفية وغير الأبجدية
    cleaned_tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    if cleaned_tokens:  # التحقق من أن النص غير فارغ بعد التنظيف
        tokenized_documents.append(cleaned_tokens)

# تحويل النصوص إلى متجهات نصية باستخدام spaCy
vectors = []
for doc_tokens in tokenized_documents:
    doc_text = " ".join(doc_tokens)
    if doc_text:  # التحقق من أن النص غير فارغ قبل إنشاء المتجه
        doc_vector = nlp(doc_text).vector
        vectors.append(doc_vector)
    else:
        vectors.append([])  # إضافة قائمة فارغة إذا كان النص فارغًا

# حفظ المتجهات النصية في ملف pkl
with open("vectors.pkl", "wb") as f:
    pickle.dump(vectors, f)
