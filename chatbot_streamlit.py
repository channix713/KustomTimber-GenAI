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
#  LOAD API KEYS (Streamlit Secrets → .env fallback)
# ================================================================

try:
    ENV_PATH = Path(__file__).parent / "OpenAIKey.env"
except NameError:
    ENV_PATH = Path(os.getcwd()) / "OpenAIKey.env"

load_dotenv(dotenv_path=ENV_PATH, override=True)

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
if not OPENAI_API_KEY:
    st.error("❗ No OpenAI API key found in st.secrets or .env")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)
MODEL = "gpt-4.1-mini"

# Load Google service credentials
GCP_JSON_STRING = st.secrets.get("GCP_SERVICE_ACCOUNT_JSON", os.getenv("GCP_SERVICE_ACCOUNT_JSON"))
if not GCP_JSON_STRING:
    st.error("❗ Missing Google service account credentials")
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
    return gc


# ================================================================
#  LOAD AND CLEAN SHEETS
# ================================================================
SPREADSHEET_ID = "1UG_N-zkgwCpObWTgmg8EPS7-N08aqintu8h3kN8yRmM"
WORKSHEET_NAME = "Stock"
WORKSHEET_NAME2 = "Summary"

@st.cache_data(show_spinner=True)
def load_sheets():
    gc = google_auth()

    # ---------------- STOCK SHEET ----------------
    stock_ws = gc.open_by_key(SPREADSHEET_ID).worksheet(WORKSHEET_NAME)
    srows = stock_ws.get_all_values()
    stock_df = pd.DataFrame(srows)
    stock_df.columns = stock_df.iloc[0].str.strip()
    stock_df = stock_df[1:].reset_index(drop=True)

    # Product Code normalization
    if "Product Code" in stock_df:
        stock_df["Product Code"] = (
            stock_df["Product Code"]
            .astype(str)
            .str.replace(r"[^0-9]", "", regex=True)
            .str.strip()
        )

    # Month Required normalization
    if "Month Required" in stock_df:
        stock_df["Month Required"] = (
            stock_df["Month Required"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        month_map = {
            "jan": "january", "feb": "february", "mar": "march", "apr": "april",
            "may": "may", "jun": "june", "jul": "july", "aug": "august",
            "sep": "september", "oct": "october", "nov": "november", "dec": "december"
        }

        def normalize_month(text):
            parts = text.split()
            if len(parts) != 2:
                return text
            m, y = parts
            m = m[:3]  # abbreviation
            if m in month_map:
                m = month_map[m]
            return f"{m} {y}"

        stock_df["Month Required"] = stock_df["Month Required"].apply(normalize_month)

    # Packs normalization
    if "Packs" in stock_df:
        stock_df["Packs_num"] = (
            stock_df["Packs"]
            .astype(str)
            .str.replace(r"[^0-9.]", "", regex=True)
        )
        stock_df["Packs_num"] = pd.to_numeric(stock_df["Packs_num"], errors="coerce")

    # Status normalization
    if "Status" in stock_df:
        stock_df["Status"] = stock_df["Status"].astype(str).str.strip().str.lower()

    # ---------------- SUMMARY SHEET ----------------
    summary_ws = gc.open_by_key(SPREADSHEET_ID).worksheet(WORKSHEET_NAME2)
    mrows = summary_ws.get_all_values()
    summary_df = pd.DataFrame(mrows)
    summary_df.columns = summary_df.iloc[0].str.strip()
    summary_df = summary_df[1:].reset_index(drop=True)

    # Clean all text columns
    for col in summary_df.columns:
        summary_df[col] = summary_df[col].astype(str).str.strip()

    # Clean numeric columns
    numeric_cols = ["AVAILABLE", "ORDERED", "LANDED", "Invoiced"]
    for col in numeric_cols:
        if col in summary_df:
            summary_df[col + "_num"] = (
                summary_df[col].str.replace(r"[^0-9.]", "", regex=True)
            )
            summary_df[col + "_num"] = pd.to_numeric(summary_df[col + "_num"], errors="coerce")

    return stock_df, summary_df


# ================================================================
#  AI QUERY FOR SELECTED SHEET
# ================================================================
def ai_query(df, question):
    cols = list(df.columns)

    prompt = f"""
Convert the user's question into a single Python expression
that runs ONLY on the pandas DataFrame named df.

DataFrame Columns:
{cols}

RULES:
- Use df["COLUMN"] syntax
- Use *_num columns for numeric operations where available
- Return ONLY Python code (no explanation, no markdown)
- Do NOT reference any other dataframe
- Do NOT merge anything
- Code must return a scalar, Series, or DataFrame

User question:
{question}
"""

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        ai_code = resp.choices[0].message.content.strip()
    except Exception as e:
        return None, None, f"OpenAI error: {e}"

    # Execute safely
    try:
        local_vars = {"df": df, "pd": pd, "np": np}
        result = eval(ai_code, {}, local_vars)
    except Exception as e:
        return ai_code, None, f"Execution error: {e}"

    result_text = (
        result.to_string() if isinstance(result, (pd.DataFrame, pd.Series)) else str(result)
    )

    return ai_code, result_text, None


# ================================================================
#  STREAMLIT UI
# ================================================================
st.title("📦 Inventory Chatbot (Manual Sheet Selection)")
st.caption("Choose stock_df or summary_df, and the AI uses only that data.")

stock_df, summary_df = load_sheets()

# Pick the sheet
sheet_choice = st.selectbox(
    "Select which sheet to query:",
    ["Stock Sheet (stock_df)", "Summary Sheet (summary_df)"]
)

df = stock_df if sheet_choice.startswith("Stock") else summary_df

if st.checkbox("Show sheet preview"):
    st.dataframe(df.head(200))

question = st.text_input("Ask a question:")

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question")
    else:
        ai_code, result_text, error = ai_query(df, question)

        if error:
            st.error(error)
        else:
            st.subheader("AI Generated Python Code")
            st.code(ai_code)

            st.subheader("Result")
            st.text(result_text)

st.markdown("---")
st.caption("🔒 Secrets loaded securely via st.secrets + .env")
