# ======================================================================
# STREAMLIT GOOGLE SHEETS CHATBOT (Stock + Summary)
# Final Clean Version — With strict validator, isolation, refresh, safe execution
# ======================================================================

import streamlit as st
import pandas as pd
import numpy as np
import gspread
import json
import re
import ast
import os
from pathlib import Path
from dotenv import load_dotenv
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials
from openai import OpenAI

# ======================================================================
# STREAMLIT SETUP
# ======================================================================
st.set_page_config(page_title="Inventory Chatbot", layout="wide")
st.title("📦 Inventory Chatbot — Stock & Summary Sheets")


# ======================================================================
# OPENAI API KEY LOADING (your required method)
# ======================================================================
try:
    ENV_PATH = Path(__file__).parent / "OpenAIKey.env"
except NameError:
    ENV_PATH = Path(os.getcwd()) / "OpenAIKey.env"

load_dotenv(dotenv_path=ENV_PATH, override=True)

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
if not OPENAI_API_KEY:
    st.error("❌ Missing OPENAI_API_KEY in secrets or .env")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)
MODEL = "gpt-4.1-mini"


# ======================================================================
# GOOGLE CREDS (your required format)
# ======================================================================
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.readonly",
]

def get_google_creds():
    """Load credentials from Streamlit Secrets."""
    if "GOOGLE_SERVICE_ACCOUNT_JSON" not in st.secrets:
        st.error("❌ Missing GOOGLE_SERVICE_ACCOUNT_JSON in secrets.")
        st.stop()
    try:
        info = json.loads(st.secrets["GOOGLE_SERVICE_ACCOUNT_JSON"])
        return Credentials.from_service_account_info(info, scopes=SCOPES)
    except Exception as e:
        st.error(f"❌ Invalid Google credentials: {e}")
        st.stop()

creds = get_google_creds()
gc = gspread.authorize(creds)
drive_service = build("drive", "v3", credentials=creds)


# ======================================================================
# SHEET CONFIG
# ======================================================================
SPREADSHEET_ID = "1UG_N-zkgwCpObWTgmg8EPS7-N08aqintu8h3kN8yRmM"
WS_STOCK = "Stock"
WS_SUMMARY = "Summary"


# ======================================================================
# AUTO-REFRESH / MANUAL REFRESH
# ======================================================================
def refresh_sheets():
    st.cache_data.clear()
    st.success("🔄 Google Sheets refreshed!")
    st.stop()

if st.button("🔄 Refresh Sheets Now"):
    refresh_sheets()


# ======================================================================
# LOAD SHEETS
# ======================================================================
@st.cache_data(show_spinner=True)
def load_sheets():

    # -------------------- STOCK --------------------
    ws = gc.open_by_key(SPREADSHEET_ID).worksheet(WS_STOCK)
    stock_df = pd.DataFrame(ws.get_all_values())
    stock_df.columns = stock_df.iloc[0].str.strip()
    stock_df = stock_df[1:].reset_index(drop=True)
    stock_df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

    for col in ["Product Code", "PACKS"]:
        if col in stock_df:
            stock_df[col] = pd.to_numeric(stock_df[col], errors="coerce")

    # -------------------- SUMMARY --------------------
    ws2 = gc.open_by_key(SPREADSHEET_ID).worksheet(WS_SUMMARY)
    summary_df = pd.DataFrame(ws2.get_all_values())
    summary_df.columns = summary_df.iloc[0].str.strip()
    summary_df = summary_df[1:].reset_index(drop=True)
    summary_df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

    numeric_cols = [
        "COST","PACK SIZE","ORDERED","LANDED","Shipped","SOH (DC)",
        "Packs (DC)","Invoiced","AVAILABLE","SOH + SOO","SOO COST","SOH COST"
    ]
    for col in numeric_cols:
        if col in summary_df:
            summary_df[col] = pd.to_numeric(summary_df[col], errors="coerce")

    # Always create AVAILABLE_num
    summary_df["AVAILABLE_num"] = pd.to_numeric(summary_df.get("AVAILABLE", np.nan), errors="coerce")

    return stock_df, summary_df


stock_df, summary_df = load_sheets()


# ======================================================================
# STRICT VALIDATOR
# ======================================================================
FORBIDDEN = [
    "import", "open(", "os.", "sys.", "subprocess", "__",
    "eval", "exec", "globals", "locals"
]

def extract_columns(code_text):
    pattern = r'\[\s*[\'"]([^\'"]+)[\'"]\s*\]'
    return re.findall(pattern, code_text)

def validate_ai_code(ai_code: str, df_columns):

    for bad in FORBIDDEN:
        if bad in ai_code:
            return False, f"Forbidden keyword: {bad}"

    if "result =" not in ai_code:
        return False, "Missing: result ="

    try:
        ast.parse(ai_code)
    except Exception as e:
        return False, f"Syntax error: {e}"

    lower_cols = [c.lower() for c in df_columns]
    for col in extract_columns(ai_code):
        if col.lower() not in lower_cols:
            return False, f"Invalid column used: {col}"

    return True, "OK"


# ======================================================================
# AI ENGINE (works per individual dataframe)
# ======================================================================
def ask_sheet(question, df_name):

    q = question.lower().strip()

    # Select dataframe + rules
    if df_name == "stock":
        df = stock_df
        df_label = "stock_df"
        id_col = "Product Code"
        value_col = "PACKS"
        forbidden = ["ITEM #", "AVAILABLE", "AVAILABLE_num"]
    else:
        df = summary_df
        df_label = "summary_df"
        id_col = "ITEM #"
        value_col = "AVAILABLE_num"
        forbidden = ["Product Code", "PACKS", "Month"]

    cols = list(df.columns)

    # --------------------- AI PROMPT ---------------------
    prompt = f"""
You are an expert pandas code generator.

You work ONLY with dataframe: {df_label}
COLUMNS = {cols}

RULES:
- Only use {df_label}
- Product ID column = "{id_col}"
- Numeric column = "{value_col}"
- Forbidden columns: {forbidden}
- NEVER use .iloc[0] → ALWAYS use .head(1)
- Final line MUST be: result = <value>
- Output ONLY python code.

QUESTION:
{q}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role":"user","content":prompt}]
    )

    ai_code = response.choices[0].message.content.strip()

    # Clean code
    ai_code = (
        ai_code.replace("```python","")
               .replace("```","")
               .replace("“",'"')
               .replace("”",'"')
               .replace("’","'")
               .strip()
    )

    # Fix .iloc[0]
    ai_code = ai_code.replace(".iloc[0]", ".head(1)")

    # VALIDATE
    ok, msg = validate_ai_code(ai_code, df.columns)
    if not ok:
        return f"❌ AI Code Validation Failed:\n{msg}\n\nCODE:\n{ai_code}"

    # EXECUTE
    try:
        local_vars = {df_label: df}
        exec(ai_code, {}, local_vars)
        result = local_vars["result"]

        if isinstance(result, (pd.DataFrame, pd.Series)) and result.empty:
            return (
                "⚠️ No matching rows found.\n\n"
                "Tried running:\n"
                f"`{ai_code}`\n\n"
                "📌 First 5 rows:\n"
                f"{df.head().to_string()}"
            )

    except Exception as e:
        return (
            f"❌ Error executing AI code: {e}\n\n"
            f"CODE:\n{ai_code}"
        )

    # FORMAT RESULT
    result_text = (
        result.to_string() if isinstance(result,(pd.DataFrame,pd.Series))
        else str(result)
    )

    # EXPLAIN RESULT
    explain_prompt = f"""
Explain clearly:

QUESTION:
{question}

RESULT:
{result_text}
"""

    explanation = client.chat.completions.create(
        model=MODEL,
        messages=[{"role":"user","content":explain_prompt}]
    ).choices[0].message.content

    return explanation


# ======================================================================
# UI
# ======================================================================
st.subheader("Choose Sheet to Query")
sheet_choice = st.selectbox("Sheet:", ["Stock Sheet", "Summary Sheet"])

df_name = "stock" if sheet_choice == "Stock Sheet" else "summary"
df_selected = stock_df if df_name == "stock" else summary_df

if st.checkbox("Show DataFrame Preview"):
    st.dataframe(df_selected, use_container_width=True)

question = st.text_input("Ask anything (case-insensitive):")

if st.button("Ask"):
    if not question.strip():
        st.warning("Enter a question.")
    else:
        st.markdown("### Chatbot Answer:")
        st.write(ask_sheet(question, df_name))
