from fastapi import FastAPI, UploadFile, File
from typing import List

from plagiarism import check_plagiarism

app = FastAPI(title="Plagiarism Detection API")


@app.post("/check-plagiarism")
async def check(
    query: UploadFile = File(...),
    references: List[UploadFile] = File(...)
):
    query_doc = (query.filename, (await query.read()).decode())

    ref_docs = [
        (ref.filename, (await ref.read()).decode())
        for ref in references
    ]

    results = check_plagiarism(query_doc, ref_docs)
    return {"results": results}
