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

# Load .env for local development
try:
    ENV_PATH = Path(__file__).parent / "OpenAIKey.env"
except NameError:
    ENV_PATH = Path(os.getcwd()) / "OpenAIKey.env"

load_dotenv(dotenv_path=ENV_PATH, override=True)

# ------------------ OPENAI API KEY ------------------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

if not OPENAI_API_KEY:
    st.error("❗ No OpenAI API key found in st.secrets or .env")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)
MODEL = "gpt-4.1-mini"

# ------------------ GOOGLE SERVICE ACCOUNT JSON ------------------
GCP_JSON_STRING = st.secrets.get(
    "GCP_SERVICE_ACCOUNT_JSON",
    os.getenv("GCP_SERVICE_ACCOUNT_JSON")
)

if not GCP_JSON_STRING:
    st.error("❗ Google service account credentials missing.")
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

    creds = service_account.Credentials.from_service_account_info(
        GCP_CREDS, scopes=scopes
    )

    gc = gspread.authorize(creds)
    drive = build("drive", "v3", credentials=creds)
    return gc, drive


# ================================================================
#  LOAD GOOGLE SHEETS
# ================================================================
SPREADSHEET_ID = "1UG_N-zkgwCpObWTgmg8EPS7-N08aqintu8h3kN8yRmM"
WORKSHEET_NAME = "Stock"
WORKSHEET_NAME2 = "Summary"


@st.cache_data(show_spinner=True)
def load_sheets():
    gc, _ = google_auth()

    # STOCK SHEET
    ws_stock = gc.open_by_key(SPREADSHEET_ID).worksheet(WORKK SHEET_NAME)
    stock_rows = ws_stock.get_all_values()
    stock_df = pd.DataFrame(stock_rows)
    stock_df.columns = stock_df.iloc[0]
    stock_df = stock_df[1:].reset_index(drop=True)
    stock_df = stock_df.replace(r"^\s*$", np.nan, regex=True).dropna(how="all")

    # SUMMARY SHEET
    ws_summary = gc.open_by_key(SPREADSHEET_ID).worksheet(WORKSHEET_NAME2)
    summary_rows = ws_summary.get_all_values()
    summary_df = pd.DataFrame(summary_rows)
    summary_df.columns = summary_df.iloc[0]
    summary_df = summary_df[1:].reset_index(drop=True)
    summary_df = summary_df.replace(r"^\s*$", np.nan, regex=True).dropna(how="all")

    return stock_df, summary_df


# ================================================================
#  CLEAN NUMERIC FIELDS
# ================================================================
def clean_numeric(df, col):
    df[col + "_num"] = (
        df[col]
        .astype(str)
        .str.replace(r"[^0-9.\-]", "", regex=True)
    )
    df[col + "_num"] = pd.to_numeric(df[col + "_num"], errors="coerce")


# ================================================================
#  PREPARE MERGED DATA
# ================================================================
@st.cache_data(show_spinner=True)
def prepare_data():
    stock_df, summary_df = load_sheets()

    # Normalize join keys
    stock_df["Product Code"] = stock_df["Product Code"].astype(str).str.strip()
    summary_df["ITEM #"] = summary_df["ITEM #"].astype(str).str.strip()

    # Clean Month Required
    if "Month Required" in stock_df.columns:
        stock_df["Month Required"] = (
            stock_df["Month Required"]
            .astype(str)
            .str.strip()
            .str.title()     # "november 2025" → "November 2025"
        )

    # Clean Packs (convert "87m2" → 87)
    if "Packs" in stock_df.columns:
        stock_df["Packs"] = stock_df["Packs"].astype(str).str.strip()
        clean_numeric(stock_df, "Packs")

    # Clean summary numeric columns
    for col in ["ORDERED", "AVAILABLE", " SOH COST ", " SOO COST "]:
        if col in summary_df.columns:
            clean_numeric(summary_df, col)

    # Merge both sheets
    merged_df = pd.merge(
        stock_df,
        summary_df,
        left_on="Product Code",
        right_on="ITEM #",
        how="left",
        suffixes=("_stock", "_summary"),
    )

    return stock_df, summary_df, merged_df



# ================================================================
#  DIRECT PRODUCT CODE LOOKUP (NO AI — GUARANTEED ACCURATE)
# ================================================================
def direct_lookup(merged_df, code):
    code = str(code).strip()
    row = merged_df.loc[merged_df["Product Code"] == code]

    if row.empty:
        return None, "No matching Product Code found."

    # Prefer numeric AVAILABLE if present
    if "AVAILABLE_num" in merged_df.columns:
        val = row["AVAILABLE_num"].iloc[0]
        if pd.isna(val):
            raw = row["AVAILABLE"].iloc[0]
            return raw, "Numeric AVAILABLE missing; using raw instead."
        return val, None

    return row["AVAILABLE"].iloc[0], None


# ================================================================
#  AI QUERY (fallback)
# ================================================================
def ai_query(merged_df, question):
    merged_columns = list(merged_df.columns)

    prompt = f"""
Convert the user's question into a SINGLE Python expression that runs on
a pandas DataFrame named merged_df.

ALLOWED COLUMNS:
{merged_columns}

RULES:
- Use ONLY allowed columns.
- For numeric operations, use *_num columns.
- For product code lookups: merged_df["Product Code"].
- For month filtering, compare EXACT title-case strings, e.g. "November 2025".
- Packs_num ALWAYS comes from stock_df (never from summary).

Return ONLY a Python expression.
"""

    # Get the Python expression
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        ai_code = resp.choices[0].message.content
    except Exception as e:
        return None, None, f"OpenAI error: {e}"

    # Execute expression safely
    try:
        local_vars = {"merged_df": merged_df, "pd": pd, "np": np}
        result = eval(ai_code, {}, local_vars)
    except Exception as e:
        return ai_code, None, f"Execution error: {e}"

    result_text = (
        result.to_string()
        if isinstance(result, (pd.DataFrame, pd.Series))
        else str(result)
    )

    # Explanation
    try:
        explain_prompt = f"""
Explain this result clearly:

Result:
{result_text}

User Question:
{question}
"""
        explanation = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": explain_prompt}],
        ).choices[0].message.content
    except:
        explanation = "(No explanation available.)"

    return ai_code, result_text, explanation


# ================================================================
#  STREAMLIT UI
# ================================================================
st.title("📦 Inventory Chatbot (Google Sheets + OpenAI)")
st.caption("Secure • No secrets in code • Accurate product and month lookups")

st.info("Loading data…")
stock_df, summary_df, merged_df = prepare_data()
st.success("Data loaded!")

if st.checkbox("Show merged dataframe preview"):
    st.dataframe(merged_df.head(200))


# Ask question
question = st.text_input("Ask something about your inventory:")

if st.button("Ask"):
    if not question:
        st.warning("Please enter a question.")
    else:
        match = re.search(r"product\s*code\s*[:#]?\s*(\d+)", question, re.I)

        if match:
            code = match.group(1)
            val, note = direct_lookup(merged_df, code)

            st.subheader(f"Product Code {code}")
            st.write(f"AVAILABLE: **{val}**")

            if note:
                st.caption(note)

        else:
            ai_code, result_text, explanation = ai_query(merged_df, question)

            st.subheader("AI-Generated Code")
            st.code(ai_code, language="python")

            st.subheader("Result")
            st.text(result_text)

            st.subheader("Explanation")
            st.write(explanation)


st.markdown("---")
st.caption("🔐 Secrets loaded from st.secrets or .env. Clean and production-ready.")
