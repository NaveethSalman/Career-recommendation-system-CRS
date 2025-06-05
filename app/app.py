import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import joblib
import pandas as pd
import numpy as np
from src.preprocess import clean_text
from src.recommender import recommend_jobs

# For reading PDF
import fitz  # PyMuPDF

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    try:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text.strip()
    except Exception as e:
        st.error(f"❌ Failed to read PDF: {e}")
        return ""

# Streamlit UI
st.set_page_config(page_title="Career Recommender", layout="centered")
st.title("Personalized Career Recommendation System")

uploaded_resume = st.file_uploader("Upload your Resume (PDF):", type=['pdf'])

resume_text = ""
if uploaded_resume is not None:
    resume_text = extract_text_from_pdf(uploaded_resume)
    if resume_text:
        st.text_area("Extracted Resume Text:", resume_text, height=300)
    else:
        st.warning("⚠️ No text could be extracted from the uploaded file.")

if st.button("🔍 Predict Career Path"):
    if not resume_text.strip():
        st.error("⚠️ Please upload a valid resume PDF.")
    else:
        try:
            model = joblib.load("models/resume_classifier.pkl")
            vec = joblib.load("models/vectorizer.pkl")

            cleaned = clean_text(resume_text)
            vec_input = vec.transform([cleaned])
            pred = model.predict(vec_input)[0]

            st.success(f"🏆 Recommended Career: {pred}")

            st.subheader("🔍 Matching Jobs:")
            try:
                jobs = recommend_jobs(cleaned, "data/Jobs_Data.csv")
                if not jobs.empty:
                    st.dataframe(jobs.reset_index(drop=True))
                else:
                    st.info("😕 No matching jobs found based on the skills.")
            except KeyError as e:
                st.error(f"⚠️ Missing expected column in dataset: {e}")
            except Exception as e:
                st.error(f"🔧 Error while recommending jobs: {e}")

        except FileNotFoundError as e:
            st.error(f"❌ Missing model or vectorizer file: {e}")
        except Exception as e:
            st.error(f"🔧 Unexpected error: {e}")

