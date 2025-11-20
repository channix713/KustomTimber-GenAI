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


# ================================================================
#  SECURE KEY LOADING
# ================================================================

try:
    ENV_PATH = Path(__file__).parent / "OpenAIKey.env"
except NameError:
    ENV_PATH = Path(os.getcwd()) / "OpenAIKey.env"

load_dotenv(dotenv_path=ENV_PATH, override=True)

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
if not OPENAI_API_KEY:
    st.error("❗ Missing OpenAI API key.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)
MODEL = "gpt-4.1-mini"

GCP_JSON_STRING = st.secrets.get("GCP_SERVICE_ACCOUNT_JSON", os.getenv("GCP_SERVICE_ACCOUNT_JSON"))
if not GCP_JSON_STRING:
    st.error("❗ Missing Google service account JSON.")
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
    ws = gc.open_by_key(SPREADSHEET_ID).worksheet(WORKSHEET_NAME)
    rows = ws.get_all_values()
    stock_df = pd.DataFrame(rows)
    stock_df.columns = stock_df.iloc[0].str.strip()
    stock_df = stock_df[1:].reset_index(drop=True)

    # Clean all text
    for col in stock_df.columns:
        stock_df[col] = stock_df[col].astype(str).str.strip()

    # Product Code cleanup
    if "Product Code" in stock_df:
        stock_df["Product Code"] = (
            stock_df["Product Code"]
            .str.replace(r"[^0-9]", "", regex=True)
            .str.strip()
        )

    # Convert Date Required → MonthNorm
    date_col = None
    for c in stock_df.columns:
        if "date" in c.lower() and "required" in c.lower():
            date_col = c
            break

    if date_col:
        stock_df[date_col] = pd.to_datetime(stock_df[date_col], errors="coerce")
        stock_df["MonthNorm"] = (
            stock_df[date_col]
            .dt.strftime("%B %Y")
            .str.lower()
        )

    # Packs cleanup
    if "Packs" in stock_df:
        stock_df["Packs_num"] = (
            stock_df["Packs"]
            .astype(str)
            .str.replace(r"[^0-9.]", "", regex=True)
        )
        stock_df["Packs_num"] = pd.to_numeric(stock_df["Packs_num"], errors="coerce")

    # Status normalization with invoice aliases
    if "Status" in stock_df:
        def normalize_status(t):
            t = t.lower().strip()
            aliases = ["inv", "invo", "invoice", "invoiced", "invc", "inv.", "invoicing"]

            for a in aliases:
                if t == a or t.startswith(a):
                    return "invoiced"
            return t

        stock_df["Status"] = stock_df["Status"].apply(normalize_status)

    # ---------------- SUMMARY SHEET ----------------
    ws2 = gc.open_by_key(SPREADSHEET_ID).worksheet(WORKSHEET_NAME2)
    rows2 = ws2.get_all_values()
    summary_df = pd.DataFrame(rows2)
    summary_df.columns = summary_df.iloc[0].str.strip()
    summary_df = summary_df[1:].reset_index(drop=True)

    for col in summary_df.columns:
        summary_df[col] = summary_df[col].astype(str).str.strip()

    # Numeric cleanup for summary sheet
    numeric_cols = ["AVAILABLE", "ORDERED", "LANDED", "Invoiced"]
    for col in numeric_cols:
        if col in summary_df:
            summary_df[col + "_num"] = (
                summary_df[col].str.replace(r"[^0-9.]", "", regex=True)
            )
            summary_df[col + "_num"] = pd.to_numeric(summary_df[col + "_num"], errors="coerce")

    return stock_df, summary_df


# ================================================================
#  AI QUERY (operates ONLY on selected dataframe)
# ================================================================
def ai_query(df, question):
    cols = list(df.columns)

    prompt = f"""
Convert the user's question into a single Python expression using ONLY 
this pandas DataFrame named df.

Columns available:
{cols}

Rules:
- Use df["column"] syntax.
- Use *_num columns for numeric fields.
- Use df["MonthNorm"] for month filtering.
- Use df["Status"] for status (already normalized).
- Only return Python code. No text, markdown, or commentary.
- Do NOT reference any other dataframe.
- Code must return a scalar, Series, or DataFrame.

User question:
{question}
"""

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ],
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

    # Format display
    result_text = (
        result.to_string() if isinstance(result, (pd.DataFrame, pd.Series)) else str(result)
    )
    return ai_code, result_text, None


# ================================================================
#  STREAMLIT UI
# ================================================================
st.title("📦 Inventory Chatbot — Stock & Summary Sheets")
st.caption("Now fully supports date-based month filtering (MonthNorm), invoice status normalization, and manual sheet selection.")

stock_df, summary_df = load_sheets()

sheet_choice = st.selectbox(
    "Select which sheet to query:",
    ["Stock Sheet (stock_df)", "Summary Sheet (summary_df)"]
)

df = stock_df if sheet_choice.startswith("Stock") else summary_df

if st.checkbox("Show sheet preview"):
    st.dataframe(df.head(200))


# DEBUG
if st.checkbox("Debug product 20373 in Stock Sheet"):
    if "Product Code" in stock_df:
        st.dataframe(stock_df[stock_df["Product Code"] == "20373"])


question = st.text_input("Ask your question:")

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        ai_code, result_text, error = ai_query(df, question)

        if error:
            st.error(error)
        else:
            st.subheader("AI-Generated Python Code")
            st.code(ai_code)

            st.subheader("Result")
            st.text(result_text)

st.markdown("---")
st.caption("🔐 Secure Secrets | ✔ MonthNorm | ✔ Status Normalized | ✔ Manual Sheet Selection")
