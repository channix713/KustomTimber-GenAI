# app.py
import os
import re
import json
import streamlit as st
import pandas as pd
import numpy as np
import gspread
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from google.oauth2 import service_account
from googleapiclient.discovery import build

# ================================================================
#  SECURE KEY LOADING (st.secrets → fallback to .env)
# ================================================================

try:
    ENV_PATH = Path(__file__).parent / "OpenAIKey.env"
except NameError:
    ENV_PATH = Path(os.getcwd()) / "OpenAIKey.env"

load_dotenv(dotenv_path=ENV_PATH, override=True)

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
if not OPENAI_API_KEY:
    st.error("❗ No OpenAI key found")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)
MODEL = "gpt-4.1-mini"


# GOOGLE CREDS
GCP_JSON_STRING = st.secrets.get("GCP_SERVICE_ACCOUNT_JSON", os.getenv("GCP_SERVICE_ACCOUNT_JSON"))
if not GCP_JSON_STRING:
    st.error("❗ Missing Google service account JSON")
    st.stop()

GCP_CREDS = json.loads(GCP_JSON_STRING)

# ================================================================
#  GOOGLE SHEETS AUTH
# ================================================================
def google_auth():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    creds = service_account.Credentials.from_service_account_info(GCP_CREDS, scopes=scopes)
    gc = gspread.authorize(creds)
    drive = build("drive", "v3", credentials=creds)
    return gc, drive

# ================================================================
#  LOAD SHEETS
# ================================================================
SPREADSHEET_ID = "1UG_N-zkgwCpObWTgmg8EPS7-N08aqintu8h3kN8yRmM"
WORKSHEET_NAME = "Stock"
WORKSHEET_NAME2 = "Summary"

@st.cache_data(show_spinner=True)
def load_sheets():
    gc, _ = google_auth()

    # --- STOCK ---
    stock_ws = gc.open_by_key(SPREADSHEET_ID).worksheet(WORKSHEET_NAME)
    srows = stock_ws.get_all_values()
    stock_df = pd.DataFrame(srows)
    stock_df.columns = stock_df.iloc[0]
    stock_df = stock_df[1:].reset_index(drop=True)
    stock_df = stock_df.replace(r"^\s*$", np.nan, regex=True)

    if "Month Required" in stock_df:
        stock_df["Month Required"] = (
            stock_df["Month Required"].astype(str).str.strip().str.title()
        )

    if "Packs" in stock_df:
        stock_df["Packs"] = stock_df["Packs"].astype(str).str.strip()
        stock_df["Packs_num"] = (
            stock_df["Packs"].astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
        )
        stock_df["Packs_num"] = pd.to_numeric(stock_df["Packs_num"], errors="coerce")

    # --- SUMMARY ---
    summary_ws = gc.open_by_key(SPREADSHEET_ID).worksheet(WORKSHEET_NAME2)
    mrows = summary_ws.get_all_values()
    summary_df = pd.DataFrame(mrows)
    summary_df.columns = summary_df.iloc[0]
    summary_df = summary_df[1:].reset_index(drop=True)
    summary_df = summary_df.replace(r"^\s*$", np.nan, regex=True)

    for col in ["AVAILABLE", "ORDERED", "LANDED", "Invoiced"]:
        if col in summary_df:
            summary_df[col + "_num"] = summary_df[col].astype(str).str.replace(
                r"[^0-9.\-]", "", regex=True
            )
            summary_df[col + "_num"] = pd.to_numeric(summary_df[col + "_num"], errors="coerce")

    return stock_df, summary_df

# ================================================================
#  DETECT WHICH SHEET TO USE
# ================================================================
def detect_sheet(question):
    q = question.lower()

    stock_keywords = [
        "pack", "packs", "month required", "job", "product code",
        "sales person", "job name"
    ]

    summary_keywords = [
        "available", "ordered", "landed", "shipped", "invoiced",
        "category", "item #"
    ]

    if any(k in q for k in stock_keywords):
        return "stock"

    if any(k in q for k in summary_keywords):
        return "summary"

    # If unclear, let AI decide:
    decision_prompt = f"""
You must answer ONLY "stock" or "summary".

Which sheet does this question refer to?

Question: "{question}"
"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": decision_prompt}]
    )
    return response.choices[0].message.content.strip().lower()


# ================================================================
#  AI QUERY FOR ONE SHEET ONLY
# ================================================================
def ai_query(df, df_name, question):
    cols = list(df.columns)

    prompt = f"""
Convert the user's question into a SINGLE Python expression that runs on:

DataFrame name: df
Columns: {cols}

RULES:
- ALWAYS return a valid Python expression using df[…]
- Use *_num columns for numeric operations
- Do NOT reference any other dataframe
- Do NOT merge sheets
- Only operate on the columns shown above

Return ONLY Python code.

User question:
{question}
"""

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        ai_code = resp.choices[0].message.content
    except Exception as e:
        return None, None, f"OpenAI error: {e}"

    # Execute expression
    try:
        local_vars = {"df": df, "pd": pd, "np": np}
        result = eval(ai_code, {}, local_vars)
    except Exception as e:
        return ai_code, None, f"Execution error: {e}"

    # Format output
    result_text = (
        result.to_string() if isinstance(result, (pd.DataFrame, pd.Series)) else str(result)
    )

    return ai_code, result_text, None


# ================================================================
#  STREAMLIT UI
# ================================================================
st.title("📦 Inventory Chatbot (Choose Sheets Dynamically)")
st.caption("Uses OpenAI + Google Sheets without merging dataframes.")

stock_df, summary_df = load_sheets()

question = st.text_input("Ask something about your inventory:")

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        sheet = detect_sheet(question)

        if sheet == "stock":
            df = stock_df
        elif sheet == "summary":
            df = summary_df
        else:
            st.error(f"Could not determine sheet: {sheet}")
            st.stop()

        st.write(f"📝 Using **{sheet}_df** for this question")

        ai_code, result_text, error = ai_query(df, sheet, question)

        if error:
            st.error(error)
        else:
            st.subheader("AI Generated Code")
            st.code(ai_code, language="python")

            st.subheader("Result")
            st.text(result_text)
