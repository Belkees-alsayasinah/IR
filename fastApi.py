from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from GetDocumentFor.Search import get_documents_for_query_dataset2, get_documents_for_query_dataset1
from QueryRefinment.QuerySuggestion import suggest_similar_queries1
from QueryRefinment.QuerySuggestion2 import suggest_similar_queries2
from TextProcessing.TextProcessing import process_text, TextProcessor
import os

app = FastAPI()
processor = TextProcessor()
template_dir = os.path.abspath('Interface')
templates = Jinja2Templates(directory=template_dir)


class QueryModel(BaseModel):
    query: str
    dataset: str


@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("Interface.html", {"request": request})


@app.post("/query")
async def process_query(query: QueryModel):
    query_text = process_text(query.query, processor)
    if query.dataset == 'dataset1':
        top_documents, cosine_similarities = get_documents_for_query_dataset1(query_text)
        similar_queries = suggest_similar_queries1(query_text, n=10)
    elif query.dataset == 'dataset2':
        top_documents, cosine_similarities = get_documents_for_query_dataset2(query_text)
        similar_queries = suggest_similar_queries2(query_text, n=10)
    else:
        return JSONResponse(content={"error": "Invalid dataset"}, status_code=400)

    top_documents_dict = top_documents.to_dict('records')
    for doc, similarity in zip(top_documents_dict, cosine_similarities):
        doc['cosine_similarity'] = similarity

    return JSONResponse(content={
        "top_documents": top_documents_dict,
        "similar_queries": [{"query": q, "qid": qid} for q, qid in similar_queries]
    })


@app.post("/suggest-query")
async def suggest_query(query: QueryModel):
    if query.dataset == 'dataset1':
        similar_queries = suggest_similar_queries1(query.query, n=10)
    elif query.dataset == 'dataset2':
        similar_queries = suggest_similar_queries2(query.query, n=10)
    else:
        return JSONResponse(content={"error": "Invalid dataset"}, status_code=400)

    suggestions = [{"label": q, "value": q} for q, qid in similar_queries]

    return JSONResponse(content={"suggestions": suggestions})


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="debug")
