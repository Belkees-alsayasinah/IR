from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from TextProcessing.TextProcessing import TextProcessor

app = FastAPI()

with open(r"C:\Users\sayas\.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\common_words.txt", 'r',
          encoding='utf-8') as file:
    words_to_remove = file.read().splitlines()

processor = TextProcessor()


class TextRequest(BaseModel):
    text: str


class TextResponse(BaseModel):
    processed_text: str


@app.post("/process_text", response_model=TextResponse)
async def process_text(request: TextRequest):
    text = request.text
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    cleaned_text = processor.cleaned_text(text)
    normalized_text = processor.normalization_example(cleaned_text)
    stemmed_text = processor.stemming_example(normalized_text)
    lemmatized_text = processor.lemmatization_example(stemmed_text)
    stopwords_removed = processor.remove_stopwords(lemmatized_text)
    numbers_converted = processor.number_to_words(stopwords_removed)
    punctuation_removed = processor.remove_punctuation(numbers_converted)
    html_removed = processor.remove_html_tags(punctuation_removed)
    contractions_expanded = processor.expand_contractions(html_removed)
    unicode_normalized = processor.normalize_unicode(contractions_expanded)
    negations_handled = processor.handle_negations(unicode_normalized)
    urls_removed = processor.remove_urls(negations_handled)

    return TextResponse(processed_text=urls_removed)


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8005)
