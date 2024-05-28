from flask import Flask, render_template, request, jsonify
from QueryRefinment.QuerySuggestion import suggest_similar_queries
from TextProcessing.TextProcessing import TextProcessor, process_text
import joblib
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
    return render_template('Interface.html')


@app.route('/query', methods=['POST'])
def process_query():
    query = request.form['query']
    query = process_text(query, processor)
    top_documents, cosine_similarities = get_documents_for_query(query, tfidf_matrix, processor, vectorizer, data)
    similar_queries = suggest_similar_queries(query, n=10)
    # Convert DataFrame to list of dictionaries
    top_documents_dict = top_documents.to_dict('records')
    for doc, similarity in zip(top_documents_dict, cosine_similarities):
        doc['cosine_similarity'] = similarity
    return jsonify(top_documents=top_documents_dict, similar_queries=[{"query": q, "qid": qid} for q, qid in similar_queries])

@app.route('/suggest-query', methods=['POST'])
def suggest_query():
    data = request.get_json()
    query = data['query']
    similar_queries = suggest_similar_queries(query, n=10)
    suggestions = [{"label": q, "value": q} for q, qid in similar_queries]
    return jsonify(suggestions=suggestions)

if __name__ == '__main__':
    app.run(debug=True)
