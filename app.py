#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# app.py (modified)
"""
Streamlit app: Google Sheets + Generative AI Data Assistant
- Spreadsheet ID is now FIXED (user cannot change it in the UI)
- Model returns answer directly (same as before, but no extra UI prompts)
"""

import os
from pathlib import Path
from typing import List

import pandas as pd
import gspread
import streamlit as st
from google.oauth2.service_account import Credentials
from openai import OpenAI
from dotenv import load_dotenv

# -----------------------
# Config / secrets
# -----------------------
try:
    ENV_PATH = Path(__file__).parent / "OpenAIKey.env"
except NameError:
    ENV_PATH = Path(os.getcwd()) / "OpenAIKey.env"

load_dotenv(dotenv_path=ENV_PATH, override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
SERVICE_ACCOUNT_FILE = "credentials.json"

# FIXED spreadsheet key
FIXED_SPREADSHEET_KEY = "1oeFZRrqr7YI52L5EAl972ghesCsBYcPiCOi09n_mCgY"  # <----- CHANGE THIS

st.set_page_config(page_title="Sheets → GenAI Assistant", layout="wide")

# -----------------------
# Load one tab from fixed spreadsheet
# -----------------------
@st.cache_data(ttl=300)
def load_fixed_sheet(worksheet_name: str = "Sheet1") -> pd.DataFrame:
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    client_gs = gspread.authorize(creds)

    sheet = client_gs.open_by_key(FIXED_SPREADSHEET_KEY).worksheet(worksheet_name)
    raw_values = sheet.get_all_values()
    if not raw_values:
        return pd.DataFrame()

    headers = raw_values[0]
    # ensure duplicate headers are unique
    unique_headers = []
    for h in headers:
        if h not in unique_headers:
            unique_headers.append(h)
        else:
            count = sum(x.startswith(h) for x in unique_headers)
            unique_headers.append(f"{h}_{count}")

    df = pd.DataFrame(raw_values[1:], columns=unique_headers)
    return df

# -----------------------
# Ask model
# -----------------------
def ask_data_question_full(question: str, df: pd.DataFrame, model_name: str = "gpt-4.1") -> str:
    if client is None:
        return "Error: OpenAI client not configured. Set OPENAI_API_KEY."

    col_map = {col: f"c{i}" for i, col in enumerate(df.columns)}
    short_df = df.rename(columns=col_map).copy()
    full_data = short_df.to_dict(orient="records")

    system_prompt = (
        "You are a senior data analyst AI with FULL access to the dataset. "
        "The dataset is provided with shortened column names and a column map. "
        "Translate shorthand names back to the original column names in your answer. "
        "Perform exact calculations and do not guess any values."
    )

    user_prompt = (
        "COLUMN MAP (short -> original):\n"
        f"{col_map}\n\n"
        "FULL DATASET (shorthand columns) as rows:\n"
        f"{full_data}\n\n"
        f"User question: {question}\n"
    )

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error: {e}"

# -----------------------
# Streamlit UI
# -----------------------
st.title("📊 Kustom Timber Stocks AI Assitant ")

with st.sidebar:
    st.header("Settings")
    worksheet_name = st.text_input("Worksheet tab name", value="Sheet1")
    model_choice = st.selectbox("Model", ["gpt-4.1", "gpt-3.5-turbo"], index=0)
    load_btn = st.button("Load Fixed Sheet")

if load_btn:
    with st.spinner("Loading sheet..."):
        try:
            df_all = load_fixed_sheet(worksheet_name)
            if df_all.empty:
                st.error("No data loaded. Check worksheet name and sharing settings.")
            else:
                st.success(f"Loaded {len(df_all)} rows, {df_all.shape[1]} columns")
                st.session_state["df_all"] = df_all
        except Exception as e:
            st.error(f"Failed to load sheet: {e}")

if "df_all" in st.session_state:
    df_all = st.session_state["df_all"]

    st.subheader("Preview (first 200 rows)")
    st.dataframe(df_all.head(200))

    # Ask question
    st.subheader("Ask a question about the data")
    question = st.text_input("Your question")

    if st.button("Ask") and question.strip():
        with st.spinner("Querying model..."):
            answer = ask_data_question_full(question, df_all, model_choice)
        st.markdown("**🤖 Answer:**")
        st.write(answer)

    st.divider()
    if st.button("Show column names"):
        st.write(list(df_all.columns))
else:
    st.info("Load the fixed spreadsheet using the sidebar.")

