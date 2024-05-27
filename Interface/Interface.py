import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from customtkinter import CTk, CTkLabel, CTkEntry, CTkButton, CTkTextbox, CTkScrollbar
from tkinter import END
from First.TextProcessing import TextProcessor, process_text

# إعداد نص المعالجة والنصوص
text_processor = TextProcessor()

# أول محاولة بحث باستخدام Tf-Idf
def process_query(query_text, processor):
    return process_text(query_text, processor)

# تحميل البيانات والمصفوفة
corpus_file = r"C:\Users\sayas\.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\collection.tsv"
df = pd.read_csv(corpus_file, delimiter='\t', header=None, names=['id', 'text'])

df = df.dropna(subset=['text'])

tfidf_matrix_file = r"C:\Users\sayas\.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\tfidf_matrix.pkl"
with open(tfidf_matrix_file, 'rb') as file:
    tfidf_matrix = joblib.load(file)

vectorizer_file = r"C:\Users\sayas\.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\tfidf_vectorizer.pkl"
with open(vectorizer_file, 'rb') as file:
    vectorizer = joblib.load(file)

# دالة البحث عن الوثائق
def search_documents(query, n=10):
    query_tfidf = vectorizer.transform([query])
    similarity_scores = cosine_similarity(tfidf_matrix, query_tfidf)
    most_relevant_doc_indices = similarity_scores.argsort(axis=0)[-n:].flatten()[::-1]
    return df.iloc[most_relevant_doc_indices]

# دالة عرض النتائج
def display_results():
    query_text = query_entry.get()
    processed_query = process_query(query_text, text_processor)
    query_tfidf = vectorizer.transform([processed_query])
    similarity_scores = cosine_similarity(tfidf_matrix, query_tfidf)

    n = 10
    most_relevant_doc_indices = similarity_scores.argsort(axis=0)[-n:].flatten()[::-1]

    threshold = 0.3
    results_textbox.delete(1.0, END)

    for i, doc_index in enumerate(most_relevant_doc_indices):
        doc_number = df.iloc[doc_index]['id']
        doc_text = df.iloc[doc_index]['text']

        query_words = processed_query.split()
        doc_words = doc_text.split()
        intersection = len(set(query_words) & set(doc_words))
        relative_similarity = intersection / len(query_words)

        if relative_similarity >= threshold:
            result_container = f"Document {doc_number} (Similarity: {similarity_scores[doc_index][0]}):\n{doc_text}\n\n"
            results_textbox.insert(END, result_container)

# إعداد واجهة المستخدم باستخدام customtkinter
root = CTk()
root.title("Document Search")

query_label = CTkLabel(root, text="Enter your query:")
query_label.pack(padx=10, pady=10)

query_entry = CTkEntry(root, width=400)
query_entry.pack(padx=10, pady=10)

search_button = CTkButton(root, text="Search", command=display_results, corner_radius=15, fg_color="green", text_color="white")
search_button.pack(padx=10, pady=10)

results_textbox = CTkTextbox(root, wrap='word', height=400, width=600, corner_radius=15)
results_textbox.pack(side="left", fill="both", expand=True, padx=10, pady=10)

scrollbar = CTkScrollbar(root, command=results_textbox.yview)
scrollbar.pack(side="right", fill="y")

results_textbox.configure(yscrollcommand=scrollbar.set)

root.mainloop()
