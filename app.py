import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/check-plagiarism"

st.title("🛡️ Plagiarism Detection System")

query = st.file_uploader("Upload Query Document", type=["txt", "py", "java", "cpp", "js"])
refs = st.file_uploader(
    "Upload Reference Documents",
    type=["txt", "py", "java", "cpp", "js"],
    accept_multiple_files=True
)

if st.button("Run Plagiarism Check"):
    if not query or not refs:
        st.error("Please upload query and reference files.")
    else:
        files = {
            "query": (query.name, query.getvalue())
        }
        ref_files = [
            ("references", (f.name, f.getvalue()))
            for f in refs
        ]

        response = requests.post(API_URL, files={**files, **dict(ref_files)})

        if response.status_code == 200:
            for r in response.json()["results"]:
                st.write(
                    f"📄 **{r['reference']}** | "
                    f"TF-IDF: {r['tfidf']} | "
                    f"SBERT: {r['sbert']} | "
                    f"🟡 {r['verdict']}"
                )
        else:
            st.error("Backend error")
