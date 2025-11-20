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
    ws_stock = gc.open_by_key(SPREADSHEET_ID).worksheet(WORKSHEET_NAME)
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

    # Clean numeric columns
    if "Packs" in stock_df.columns:
        clean_numeric(stock_df, "Packs")

    for col in ["ORDERED", "AVAILABLE", " SOH COST ", " SOO COST "]:
        if col in summary_df.columns:
            clean_numeric(summary_df, col)

    # FIXED — .str.strip() instead of .strip()
    stock_df["Product Code"] = stock_df["Product Code"].astype(str).str.strip()
    summary_df["ITEM #"] = summary_df["ITEM #"].astype(str).str.strip()

    # Merge sheets
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
#  DIRECT PRODUCT CODE LOOKUP (NO AI — ALWAYS CORRECT)
# ================================================================
def direct_lookup(merged_df, code):
    code = str(code).strip()
    row = merged_df.loc[merged_df["Product Code"] == code]

    if row.empty:
        return None, "No matching Product Code found."

    if "AVAILABLE_num" in merged_df.columns:
        val = row["AVAILABLE_num"].iloc[0]
        if pd.isna(val):
            raw = row["AVAILABLE"].iloc[0]
            return raw, "Converted numeric unavailable; using raw AVAILABLE value."
        return val, None

    return row["AVAILABLE"].iloc[0], None


# ================================================================
#  AI QUERY (fallback)
# ================================================================
def ai_query(merged_df, question):
    merged_columns = list(merged_df.columns)

    prompt = f"""
You are a senior data analyst. Convert the user's question into a single Python
expression operating on the DataFrame 'merged_df'.

Allowed Columns:
{merged_columns}

RULES:
- Use *_num columns for numeric operations.
- For product code lookups, ALWAYS use merged_df["Product Code"].
- Return ONLY Python code (no commentary, no markdown).
"""

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        ai_code = resp.choices[0].message.content
    except Exception as e:
        return None, None, f"OpenAI error: {e}"

    # Execute the AI-generated code
    try:
        local_vars = {"merged_df": merged_df, "pd": pd, "np": np}
        result = eval(ai_code, {}, local_vars)
    except Exception as e:
        return ai_code, None, f"Execution error: {e}"

    if isinstance(result, (pd.Series, pd.DataFrame)):
        result_text = result.to_string()
    else:
        result_text = str(result)

    # Generate explanation
    try:
        explain_prompt = f"""
Explain this result clearly:

Result:
{result_text}

Question:
{question}
"""
        explain_resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": explain_prompt}],
        )
        explanation = explain_resp.choices[0].message.content
    except:
        explanation = "(No explanation available.)"

    return ai_code, result_text, explanation


# ================================================================
#  STREAMLIT UI
# ================================================================
st.title("📦 Inventory Chatbot (Google Sheets + OpenAI)")
st.caption("Secure • No API keys stored in code • Uses Streamlit Secrets + .env")

st.info("Loading data from Google Sheets…")
stock_df, summary_df, merged_df = prepare_data()
st.success("Sheets loaded successfully!")

if st.checkbox("Show merged dataframe preview"):
    st.dataframe(merged_df.head(200))


# User Question
question = st.text_input("Ask something about your inventory:")

if st.button("Ask"):
    if not question:
        st.warning("Please enter a question.")
    else:
        # Detect "Product Code 12345"
        match = re.search(r"product\s*code\s*[:#]?\s*(\d+)", question, re.I)

        if match:
            code = match.group(1)
            val, note = direct_lookup(merged_df, code)

            st.subheader(f"Product Code {code}")
            st.write(f"AVAILABLE: **{val}**")

            if note:
                st.caption(note)

        else:
            # AI fallback
            ai_code, result_text, explanation = ai_query(merged_df, question)

            st.subheader("AI-Generated Python Code")
            st.code(ai_code, language="python")

            st.subheader("Result")
            st.text(result_text)

            st.subheader("Explanation")
            st.write(explanation)


st.markdown("---")
st.caption("🔐 Secrets loaded securely via st.secrets & .env.")
