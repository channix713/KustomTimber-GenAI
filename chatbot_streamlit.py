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
#  SECURE KEY LOADING
# ================================================================

# Load local .env if available
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

# Google service account
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
    return gc


# ================================================================
#  LOAD SHEETS
# ================================================================
SPREADSHEET_ID = "1UG_N-zkgwCpObWTgmg8EPS7-N08aqintu8h3kN8yRmM"
WORKSHEET_NAME = "Stock"
WORKSHEET_NAME2 = "Summary"

@st.cache_data(show_spinner=True)
def load_sheets():
    gc = google_auth()

    # ---- STOCK ----
    stock_ws = gc.open_by_key(SPREADSHEET_ID).worksheet(WORKSHEET_NAME)
    stock_rows = stock_ws.get_all_values()
    stock_df = pd.DataFrame(stock_rows)
    stock_df.columns = stock_df.iloc[0]
    stock_df = stock_df[1:].reset_index(drop=True)
    stock_df = stock_df.replace(r"^\s*$", np.nan, regex=True)

    # Normalize Month Required
    if "Month Required" in stock_df:
        stock_df["Month Required"] = (
            stock_df["Month Required"].astype(str).str.strip().str.title()
        )

    # Clean Packs
    if "Packs" in stock_df:
        stock_df["Packs"] = stock_df["Packs"].astype(str).str.strip()
        stock_df["Packs_num"] = (
            stock_df["Packs"].str.replace(r"[^0-9.\-]", "", regex=True)
        )
        stock_df["Packs_num"] = pd.to_numeric(stock_df["Packs_num"], errors="coerce")

    # ---- SUMMARY ----
    summary_ws = gc.open_by_key(SPREADSHEET_ID).worksheet(WORKSHEET_NAME2)
    summary_rows = summary_ws.get_all_values()
    summary_df = pd.DataFrame(summary_rows)
    summary_df.columns = summary_df.iloc[0]
    summary_df = summary_df[1:].reset_index(drop=True)
    summary_df = summary_df.replace(r"^\s*$", np.nan, regex=True)

    # Clean numeric summary columns
    for col in ["AVAILABLE", "ORDERED", "LANDED", "Invoiced"]:
        if col in summary_df:
            summary_df[col + "_num"] = (
                summary_df[col].astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
            )
            summary_df[col + "_num"] = pd.to_numeric(summary_df[col + "_num"], errors="coerce")

    return stock_df, summary_df


# ================================================================
#  AI QUERY FOR SELECTED SHEET
# ================================================================
def ai_query(df, question):
    cols = list(df.columns)

    prompt = f"""
Convert the user's question into a SINGLE Python expression using ONLY
this pandas DataFrame named df.

DataFrame Columns:
{cols}

RULES:
- Use df["COLUMN"] to reference data
- Use *_num columns for numeric fields
- Return ONLY Python code (no markdown, no explanation)
- Do NOT reference other dataframes
- Do NOT merge dataframes
- Your code must return a scalar, Series, or DataFrame

User question:
{question}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        ai_code = response.choices[0].message.content.strip()
    except Exception as e:
        return None, None, f"OpenAI Error: {e}"

    # Execute the code
    try:
        local_vars = {"df": df, "pd": pd, "np": np}
        result = eval(ai_code, {}, local_vars)
    except Exception as e:
        return ai_code, None, f"Execution Error: {e}"

    result_text = (
        result.to_string() if isinstance(result, (pd.DataFrame, pd.Series)) else str(result)
    )

    return ai_code, result_text, None


# ================================================================
#  STREAMLIT UI
# ================================================================
st.title("📦 Inventory Chatbot (Manual Sheet Selection)")
st.caption("You choose the sheet → AI answers using only that data.")

stock_df, summary_df = load_sheets()

# User selects data source
sheet_choice = st.selectbox(
    "Select which sheet to query:",
    ["Stock Sheet (stock_df)", "Summary Sheet (summary_df)"]
)

df = stock_df if sheet_choice.startswith("Stock") else summary_df

st.write(f"Using **{sheet_choice}**")

if st.checkbox("Show sheet preview"):
    st.dataframe(df.head(200))

question = st.text_input("Ask a question about this sheet:")

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        ai_code, result_text, error = ai_query(df, question)

        if error:
            st.error(error)
        else:
            st.subheader("AI Generated Code")
            st.code(ai_code, language="python")

            st.subheader("Result")
            st.text(result_text)

st.markdown("---")
st.caption("🔐 All secrets securely loaded using st.secrets + .env")
