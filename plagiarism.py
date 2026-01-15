import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


TEXT_EXTENSIONS = {".txt", ".md"}
CODE_EXTENSIONS = {".py", ".java", ".cpp", ".js"}

TFIDF_THRESHOLD = 0.75
SBERT_THRESHOLD = 0.80
MODEL_NAME = "all-MiniLM-L6-v2"


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def preprocess_code(code: str) -> str:
    code = re.sub(r"#.*", "", code)
    code = re.sub(r"//.*", "", code)
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.S)
    code = re.sub(r"\s+", " ", code)
    return code.strip().lower()


def preprocess(name: str, content: str) -> str:
    return preprocess_code(content) if Path(name).suffix in CODE_EXTENSIONS else preprocess_text(content)


def check_plagiarism(
    query: Tuple[str, str],
    references: List[Tuple[str, str]]
):
    docs = [query] + references
    processed = [preprocess(n, c) for n, c in docs]

    tfidf = TfidfVectorizer(ngram_range=(1, 3), stop_words="english").fit_transform(processed)
    tfidf_sim = cosine_similarity(tfidf)

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(processed, normalize_embeddings=True)
    sbert_sim = cosine_similarity(embeddings)

    results = []
    for i, (ref_name, _) in enumerate(references, start=1):
        results.append({
            "reference": ref_name,
            "tfidf": round(float(tfidf_sim[0][i]), 3),
            "sbert": round(float(sbert_sim[0][i]), 3),
            "verdict": "Suspicious" if (
                tfidf_sim[0][i] >= TFIDF_THRESHOLD or
                sbert_sim[0][i] >= SBERT_THRESHOLD
            ) else "Clean"
        })

    return results
