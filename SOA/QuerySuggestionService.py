from fastapi import FastAPI
from pydantic import BaseModel
from QueryRefinment.QuerySuggestion import suggest_similar_queries1
from QueryRefinment.QuerySuggestion2 import suggest_similar_queries2

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    dataset: str

@app.post("/suggest_query_result")
def suggest_query_endpoint(request: QueryRequest):
    if request.dataset == 'dataset1':
        suggestions = suggest_similar_queries1(request.query, n=10)
    elif request.dataset == 'dataset2':
        suggestions = suggest_similar_queries2(request.query, n=10)
    else:
        return {"error": "Invalid dataset"}
    return {"suggestions": suggestions}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)