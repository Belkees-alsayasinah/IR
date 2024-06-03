from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import requests
import os

app = FastAPI()
template_dir = os.path.abspath('Interface')
templates = Jinja2Templates(directory=template_dir)


class QueryModel(BaseModel):
    query: str
    dataset: str


async def process_text_via_api(query):
    url = "http://127.0.0.1:8005/process_text"
    payload = {"text": query}
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=payload, headers=headers)

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    return response.json()["processed_text"]


async def get_documents_via_api(query, dataset):
    if dataset == 'dataset1':
        url = "http://127.0.0.1:8006/retrieve_documents_dataset1"
    elif dataset == 'dataset2':
        url = "http://127.0.0.1:8007/retrieve_documents_dataset2"
    else:
        raise HTTPException(status_code=400, detail="Invalid dataset")

    payload = {"query": query}
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=payload, headers=headers)

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    return response.json()


async def get_Suggestion_Query(query, dataset):
    url = "http://127.0.0.1:8008/suggest_query_result"

    if dataset not in ['dataset1', 'dataset2']:
        raise HTTPException(status_code=400, detail="Invalid dataset")

    payload = {"query": query, "dataset": dataset}
    headers = {"Content-Type": "application/json"}

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    return response.json()


@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("Interface.html", {"request": request})


@app.post("/query")
async def process_query(query: QueryModel):
    query_text = await process_text_via_api(query.query)
    documents_response = await get_documents_via_api(query_text, query.dataset)

    top_documents = documents_response['documents']
    cosine_similarities = documents_response['similarities']

    similar_queries_response = await get_Suggestion_Query(query_text, query.dataset)
    similar_queries = similar_queries_response["suggestions"]

    for doc, similarity in zip(top_documents, cosine_similarities):
        doc['cosine_similarity'] = similarity

    return JSONResponse(content={
        "top_documents": top_documents,
        "similar_queries": similar_queries
    })


@app.post("/suggest-query")
async def suggest_query(query: QueryModel):
    similar_queries_response = await get_Suggestion_Query(query.query, query.dataset)
    similar_queries = similar_queries_response["suggestions"]

    if query.dataset not in ['dataset1', 'dataset2']:
        return JSONResponse(content={"error": "Invalid dataset"}, status_code=400)

    suggestions = [{"label": q, "value": q} for q, qid in similar_queries]

    return JSONResponse(content={"suggestions": suggestions})


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8081, log_level="debug")
