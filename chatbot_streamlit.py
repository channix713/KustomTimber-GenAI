# ======================================================================
# STREAMLIT GOOGLE SHEETS CHATBOT (Stock + Summary)
# With get_google_creds(), strict validator, auto-refresh, case-insensitive
# ======================================================================

import streamlit as st
import pandas as pd
import numpy as np
import gspread
import os
import json
import re
import ast
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
# LOAD OPENAI KEY (your required method)
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
# GOOGLE CREDS
# ======================================================================
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.readonly",
]

def get_google_creds():
    """Load credentials from Streamlit Secrets."""
    if "GOOGLE_SERVICE_ACCOUNT_JSON" not in st.secrets:
        st.error("❌ GOOGLE_SERVICE_ACCOUNT_JSON missing in secrets.")
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
    st.success("🔄 Google Sheets refreshed successfully!")
    st.stop()

if st.button("🔄 Refresh Sheets Now"):
    refresh_sheets()


# ======================================================================
# LOAD SHEETS
# ======================================================================
@st.cache_data(show_spinner=True)
def load_sheets():
    # STOCK
    ws = gc.open_by_key(SPREADSHEET_ID).worksheet(WS_STOCK)
    stock_df = pd.DataFrame(ws.get_all_values())
    stock_df.columns = stock_df.iloc[0].str.strip()
    stock_df = stock_df[1:].reset_index(drop=True)
    stock_df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

    # Convert numeric fields
    for col in ["Product Code", "PACKS"]:
        if col in stock_df:
            stock_df[col] = pd.to_numeric(stock_df[col], errors="coerce")

    # SUMMARY
    ws2 = gc.open_by_key(SPREADSHEET_ID).worksheet(WS_SUMMARY)
    summary_df = pd.DataFrame(ws2.get_all_values())
    summary_df.columns = summary_df.iloc[0].str.strip()
    summary_df = summary_df[1:].reset_index(drop=True)
    summary_df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

    summary_numeric = [
        "COST","PACK SIZE","ORDERED","LANDED","Shipped","SOH (DC)","Packs (DC)",
        "Invoiced","AVAILABLE","SOH + SOO","SOO COST","SOH COST"
    ]
    for col in summary_numeric:
        if col in summary_df:
            summary_df[col] = pd.to_numeric(summary_df[col], errors="coerce")

    return stock_df, summary_df


stock_df, summary_df = load_sheets()


# ======================================================================
# STRICT VALIDATOR
# ======================================================================
FORBIDDEN = ["import", "open(", "os.", "sys.", "subprocess", "__", "eval", "exec"]

def extract_columns(code_text):
    """Extract columns used inside df['col'] or df["col"] patterns."""
    pattern = r'\[\s*[\'"]([^\'"]+)[\'"]\s*\]'
    return re.findall(pattern, code_text)

def validate_ai_code(ai_code: str, df_columns):
    """Strict security + correctness validation."""
    # Block forbidden keywords
    for bad in FORBIDDEN:
        if bad in ai_code:
            return False, f"Forbidden keyword detected: {bad}"

    # Must define result =
    if not re.search(r"result\s*=", ai_code):
        return False, "Missing required: result ="

    # Python syntax validation
    try:
        ast.parse(ai_code)
    except Exception as e:
        return False, f"Syntax error: {e}"

    # Column validation (case-insensitive)
    df_cols_lower = [c.lower() for c in df_columns]
    for col in extract_columns(ai_code):
        if col.lower() not in df_cols_lower:
            return False, f"Invalid column used: '{col}'"

    return True, "OK"


# ======================================================================
# AI ENGINE
# ======================================================================
def ask_sheet(question, df_name):

    # CASE-INSENSITIVE NORMALIZATION
    question_normalized = question.lower()

    df = stock_df if df_name == "stock" else summary_df
    cols = list(df.columns)

    prompt = f"""
You are an expert pandas code generator.

DATAFRAME = {df_name}_df
COLUMNS = {cols}

STRICT RULES:
1. Only use EXACT column names shown above (case-insensitive allowed).
2. For date/month filters → use 'Month'
3. DO NOT use 'Month Required'
4. For pack-related questions → always use 'PACKS'
5. Return ONLY valid python code.
6. Final line must set:
       result = <value>

USER QUESTION:
{question_normalized}
"""

    # Ask AI
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    ai_code = resp.choices[0].message.content.strip()

    # Sanitize quotes + remove fenced code
    ai_code = ai_code.replace("“", '"').replace("”", '"').replace("’", "'")
    ai_code = ai_code.replace("```python", "").replace("```", "")

    # Replace Month Required hallucination
    ai_code = ai_code.replace("['Month Required']", "['Month']")

    # VALIDATE CODE
    ok, msg = validate_ai_code(ai_code, df.columns)
    if not ok:
        return f"❌ AI Code Validation Failed:\n{msg}\n\nCODE:\n{ai_code}"

    # EXECUTE CODE
    try:
        local_vars = {
            "stock_df": stock_df,
            "summary_df": summary_df,
            df_name + "_df": df
        }
        exec(ai_code, {}, local_vars)
        result = local_vars["result"]
    except Exception as e:
        return f"❌ Error executing AI code: {e}\n\nCODE:\n{ai_code}"

    # Format result
    result_text = (
        result.to_string() if isinstance(result, (pd.DataFrame, pd.Series)) else str(result)
    )

    # Ask AI to explain result
    explain_prompt = f"""
Explain this result clearly.

QUESTION:
{question}

RESULT:
{result_text}
"""
    explanation = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": explain_prompt}]
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

question = st.text_input("Ask your question (case-insensitive):")

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        st.markdown("### Chatbot Answer:")
        st.write(ask_sheet(question, df_name))
