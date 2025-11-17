#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# app.py
"""
Streamlit app: Google Sheets + Generative AI Data Assistant
Author: Converted from your working script
Notes:
 - Expects Google service account JSON at credentials.json (or change SERVICE_ACCOUNT_FILE)
 - Expects OpenAI API key in OpenAIKey.env OR environment variable OPENAI_API_KEY
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

# Load .env locally if present
load_dotenv(dotenv_path=ENV_PATH, override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client lazily to avoid errors during import when key missing
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
SERVICE_ACCOUNT_FILE = "credentials.json"  # put your service account file here

st.set_page_config(page_title="Sheets → GenAI Assistant", layout="wide")

# -----------------------
# Helper: load sheets (cached)
# -----------------------
@st.cache_data(ttl=300)
def load_all_sheets(sheet_keys: List[str], worksheet_name: str = "Sheet1") -> pd.DataFrame:
    """
    Load one tab (worksheet_name) from each spreadsheet key and return a concatenated DataFrame.
    Cached to avoid reloading on every interaction.
    """
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    client_gs = gspread.authorize(creds)
    df_list = []

    for key in sheet_keys:
        sheet = client_gs.open_by_key(key).worksheet(worksheet_name)
        raw_values = sheet.get_all_values()
        if not raw_values:
            continue
        headers = raw_values[0]
        # make sure duplicate headers become unique
        unique_headers = []
        for h in headers:
            if h not in unique_headers:
                unique_headers.append(h)
            else:
                # append a running index if header repeats
                count = sum(x.startswith(h) for x in unique_headers)
                unique_headers.append(f"{h}_{count}")
        df = pd.DataFrame(raw_values[1:], columns=unique_headers)
        df["Source File"] = key
        df_list.append(df)

    if not df_list:
        return pd.DataFrame()
    df_all = pd.concat(df_list, ignore_index=True)
    return df_all

# -----------------------
# Helper: call model (full row-level data with short column names)
# -----------------------
def ask_data_question_full(question: str, df: pd.DataFrame, model_name: str = "gpt-4.1") -> str:
    """
    Send the full dataset (with shortened column names) to the model.
    Returns the model's raw text response.
    """
    if client is None:
        return "Error: OpenAI client not configured. Set OPENAI_API_KEY in OpenAIKey.env or environment."

    # Build short column map to save tokens
    col_map = {col: f"c{i}" for i, col in enumerate(df.columns)}
    short_df = df.rename(columns=col_map).copy()
    full_data = short_df.to_dict(orient="records")

    system_prompt = (
        "You are a senior data analyst AI with FULL access to the dataset. "
        "The dataset is provided with shortened column names and a column map. "
        "Use the column map to translate shorthand names back to the original descriptive column names "
        "in your answer. You can filter, group, aggregate, and compute values. Show calculations when helpful. "
        "Do not invent or guess values that are not present."
    )

    user_prompt = (
        "COLUMN MAP (short -> original):\n"
        f"{col_map}\n\n"
        "FULL DATASET (shorthand columns), as a list of rows:\n"
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
        return f"⚠️ Error while querying AI: {e}"

# -----------------------
# Streamlit UI
# -----------------------
st.title("📊 Google Sheets + GenAI Data Assistant (Streamlit)")

with st.sidebar:
    st.header("Settings / Load Data")
    sheet_keys_text = st.text_area(
        "Spreadsheet keys (one per line)",
        value="",
        height=120,
        help="Paste the Google Sheets ID(s) — the long string in the sheet URL."
    )
    worksheet_name = st.text_input("Worksheet tab name (case-sensitive)", value="Sheet1")
    model_choice = st.selectbox("Model to use", options=["gpt-4.1", "gpt-3.5-turbo"], index=0)
    load_data_btn = st.button("Load sheet(s)")

# Prepare keys list
sheet_keys = [k.strip() for k in sheet_keys_text.splitlines() if k.strip()]

# Load data when button clicked
if load_data_btn:
    if not sheet_keys:
        st.sidebar.error("Please paste at least one spreadsheet key.")
    else:
        with st.spinner("Loading sheet(s)..."):
            try:
                df_all = load_all_sheets(sheet_keys, worksheet_name=worksheet_name)
                if df_all.empty:
                    st.error("No data loaded. Check keys, worksheet name and service account sharing.")
                else:
                    st.success(f"Loaded {len(df_all)} rows, columns: {df_all.shape[1]}")
                    st.session_state["df_all"] = df_all
            except Exception as e:
                st.error(f"Failed to load sheets: {e}")

# If data is loaded show interfaces
if "df_all" in st.session_state:
    df_all = st.session_state["df_all"]

    st.subheader("Preview (first 200 rows)")
    st.dataframe(df_all.head(200))

    # Optional auto-convert numeric-like columns
    if st.checkbox("Auto-convert numeric-like columns (try to parse numbers)", value=True):
        converted = False
        for col in df_all.columns:
            sample = df_all[col].astype(str).str.replace(r'[^0-9\.-]', '', regex=True)
            # fraction of non-empty numeric-like strings
            non_empty = sample.replace('', pd.NA).dropna()
            numeric_fraction = 0
            if len(non_empty) > 0:
                numeric_fraction = non_empty.str.replace('-', '').str.replace('.', '').str.isnumeric().sum() / len(non_empty)
            if numeric_fraction > 0.5:
                df_all[col] = pd.to_numeric(sample, errors='coerce')
                converted = True
        if converted:
            st.info("Attempted numeric conversions on likely numeric columns.")
        st.session_state["df_all"] = df_all

    st.subheader("Ask a question")
    question = st.text_input("Type your question (e.g., sum of Stock Count for PRODUCT CODE TEL-20211)", key="q_input")

    if st.button("Ask model") and question.strip():
        with st.spinner("Querying model..."):
            answer = ask_data_question_full(question.strip(), df_all, model_name=model_choice)
            st.markdown("**🤖 AI Answer:**")
            st.write(answer)

    st.divider()
    st.markdown("**Quick actions**")
    if st.button("Show column names"):
        st.write(list(df_all.columns))

    if st.button("Show sample as JSON"):
        st.json(df_all.head(20).to_dict(orient="records"))

else:
    st.info("Load your spreadsheet(s) using the sidebar: paste sheet key(s) and click 'Load sheet(s)'.")

