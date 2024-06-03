import sys
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()


class QueryRequest(BaseModel):
    query: str


class DocumentResponse(BaseModel):
    documents: list
    similarities: list


def load_dataset(file_path):
    try:
        data = pd.read_csv(file_path, delimiter='\t', header=None, names=['pid', 'text'])
    except pd.errors.ParserError as e:
        print(f"Error reading the dataset file: {e}")
        sys.exit(1)
    return data


def process_text_via_api(query):
    url = "http://127.0.0.1:8005/process_text"
    payload = {"text": query}
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=payload, headers=headers)

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    return response.json()["processed_text"]

@app.post("/retrieve_documents_dataset2", response_model=DocumentResponse)
async def retrieve_documents_dataset2(request: QueryRequest):
    query = request.query
    dataset2_path = r'C:\Users\sayas\.ir_datasets\antique\collection.tsv'
    tfidf_matrix2_file = r"C:\Users\sayas\.ir_datasets\antique\Final\tfidf_matrixF.pkl"
    vectorizer2_file = r"C:\Users\sayas\.ir_datasets\antique\Final\tfidf_vectorF.pkl"

    data2 = load_dataset(dataset2_path)
    data2.dropna(subset=['text'], inplace=True)

    with open(tfidf_matrix2_file, 'rb') as file:
        tfidf_matrix2 = joblib.load(file)

    with open(vectorizer2_file, 'rb') as file:
        vectorizer2 = joblib.load(file)

    processed_query = process_text_via_api(query)
    query_vector = vectorizer2.transform([processed_query])
    cosine_similarities = cosine_similarity(tfidf_matrix2, query_vector).flatten()

    n = 10
    top_documents_indices = cosine_similarities.argsort()[-n:][::-1]
    if any(idx >= len(data2) for idx in top_documents_indices):
        raise HTTPException(status_code=500, detail="Top document indices out-of-bounds")

    top_documents = data2.iloc[top_documents_indices].to_dict(orient='records')

    return DocumentResponse(documents=top_documents, similarities=cosine_similarities[top_documents_indices].tolist())
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8007)
