from flask import Flask, render_template, request, jsonify
from First.TextProcessing import TextProcessor, process_text
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from R.oj import get_documents_for_query, load_dataset

dataset_path = r'C:\Users\sayas\.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\collection.tsv'
data = load_dataset(dataset_path)
data.dropna(subset=['text'], inplace=True)

tfidf_matrix_file = r"C:\Users\sayas\.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\tfidf_matrix.pkl"
with open(tfidf_matrix_file, 'rb') as file:
    tfidf_matrix = joblib.load(file)

vectorizer_file = r"C:\Users\sayas\.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\tfidf_vectorizer.pkl"
with open(vectorizer_file, 'rb') as file:
    vectorizer = joblib.load(file)

processor = TextProcessor()
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/query', methods=['POST'])
def process_query():
    query = request.form['query']
    query = process_text(query, processor)
    top_documents, cosine_similarities = get_documents_for_query(query, tfidf_matrix, processor, vectorizer, data)
    # Convert DataFrame to list of dictionaries
    top_documents_dict = top_documents.to_dict('records')
    print(top_documents_dict)
    return jsonify(top_documents=top_documents_dict)


if __name__ == '__main__':
    app.run(debug=True)
